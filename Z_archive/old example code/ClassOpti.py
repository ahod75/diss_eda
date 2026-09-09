import pandas as pd
import pyomo.environ as pyo
import numpy as np
import random

from datetime import datetime, timedelta
from functools import reduce

class Opti:
    def __init__(self, number_houses):
        self.houses = ['house'+str(i) for i in np.arange(number_houses)]
        #Dicts and dataframes for saving results
        self.keys_estimate_soe = ['delta_g', 'g', 'p', 'p_minus', 'p_plus', 'e']
        self.keys_offline = ['g', 'g_minus', 'g_plus', 'p', 'p_minus', 'p_plus', 'e']
        self.keys_online = ['delta_g', 'g', 'p', 'p_minus', 'p_plus', 'e']
        self.results_offline = {key : {} for key in self.keys_offline}
        self.results_online = {key : {} for key in self.keys_online}
        self.results_estimate_soe = {key : {} for key in self.keys_estimate_soe}
        self.results_offline_final = {key1: {key2: pd.DataFrame() for key2 in self.houses} for key1 in self.keys_offline}
        self.results_estimate_soe_final = {key1: {key2: pd.DataFrame() for key2 in self.houses} for key1 in self.keys_online}
        self.results_online_final = {key1: {key2: pd.DataFrame() for key2 in self.houses} for key1 in self.keys_online}
        #Parameter for scheduling
        self.hour_start_computation = 12
        self.hour_start_schedule = 0
        self.end_data_schedule = datetime(2022, 12, 27)
        #Parameters for optimisation problem
        self.c_lin_plus = 0.3
        self.c_quadr_plus = 0.05
        self.c_lin_minus = 0.15
        self.c_quadr_minus = 0.05
        self.mu = {'house'+str(i): 0.05 for i in np.arange(number_houses)}
        self.relaxation_p = {'house'+str(i): 1e-8 for i in np.arange(number_houses)}
        self.p_min = {'house'+str(i): -5 for i in np.arange(number_houses)}
        self.p_max = {'house'+str(i): 5 for i in np.arange(number_houses)}
        self.e_min = {'house'+str(i): 0 for i in np.arange(number_houses)}
        self.e_max = {'house'+str(i): 13.5 for i in np.arange(number_houses)}
        self.delta = 1
        self.g_max = 500
    

    def compute_ds_deterministic(self, estimated_e, forecast_l, count_error = 0):
        model = pyo.ConcreteModel()

        # Define & initialise input parameter e, l, and DS
        model.indices = pyo.Set(dimen=2, initialize=list(forecast_l.keys()))
        model.indices_houses = pyo.Set(initialize=np.unique([j[0] for j in forecast_l.keys()]))
        model.indices_time = pyo.Set(initialize=[j[1] for j in forecast_l.keys()])
        model.indices_time_e = pyo.Set(
            initialize=[j[1] for j in forecast_l.keys()] + [[j[1] + timedelta(hours=1) for j in forecast_l.keys()][-1]],
            ordered=True)
        model.e0 = pyo.Param(pyo.Set(dimen=2, initialize=list(estimated_e.keys())), initialize=estimated_e)
        model.l = pyo.Param(model.indices, initialize=forecast_l)

        # Define decision variables
        self.define_variables(model)

        def bounds_g_plus(model, i, j):
            return (0, self.g_max)

        model.g_plus = pyo.Var(model.indices, bounds=bounds_g_plus)

        def bounds_g_minus(model, i, j):
            return (-self.g_max, 0)

        model.g_minus = pyo.Var(model.indices, bounds=bounds_g_minus)

        # Define & initialise objective
        def objective(model):
            return sum((self.c_lin_plus * model.g_plus[i] + self.c_quadr_plus * (model.g_plus[i]) ** 2 + \
                        self.c_lin_minus * model.g_minus[i] + self.c_quadr_minus * (model.g_minus[i]) ** 2) for i in
                       model.indices)

        model.objective = pyo.Objective(rule=objective, sense=pyo.minimize)

        # Define & initialise constraints
        ##g = g_plus + g_minus
        def constr_split_g(model, i, j):
            return model.g[i, j] == model.g_plus[i, j] + model.g_minus[i, j]

        model.constr_split_g = pyo.Constraint(model.indices, rule=constr_split_g)

        ##All other constraints
        self.define_constraints(model)

        # Solve optimisation problem
        optimizer = pyo.SolverFactory('ipopt')
        optimizer.options['max_iter'] = 5000
        try:
            optimizer.solve(model)
            return(model)
        except ValueError as v:
            count_error = count_error + 1
            print(v, count_error)
            if count_error <= 5:
                estimated_e_temp = {key: round(value + random.random()/1000, 4) for key, value in estimated_e.items()}
                return(self.compute_ds_deterministic(estimated_e_temp, forecast_l, count_error))

  

    def controlled_ess(self, actual_e, DS, actual_l, count_error=0):
        model = pyo.ConcreteModel()

        # Define & initialise changing parameter e, l, and DS
        model.indices = pyo.Set(dimen=2, initialize=list(actual_l.keys()))
        # list(zip(*actual_l.keys()))[1]
        model.indices_houses = pyo.Set(initialize=np.unique([j[0] for j in actual_l.keys()]))
        model.indices_time = pyo.Set(initialize=[j[1] for j in actual_l.keys()])
        model.indices_time_e = pyo.Set(
            initialize=[j[1] for j in actual_l.keys()] + [[j[1] + timedelta(hours=1) for j in actual_l.keys()][-1]],
            ordered=True)
        model.e0 = pyo.Param(pyo.Set(dimen=2, initialize=list(actual_e.keys())), initialize=actual_e)
        model.l = pyo.Param(model.indices, initialize=actual_l)
        model.DS = pyo.Param(model.indices, initialize=DS)

        # Define decision variables
        model.delta_g = pyo.Var(model.indices)
        self.define_variables(model)

        # Define & initialise objective
        def objective(model):
            return sum(model.delta_g[i] ** 2 for i in model.indices)

        model.objective = pyo.Objective(rule=objective, sense=pyo.minimize)

        # Define & initialise constraints
        ##g = DS + delta_g
        def constr_actual_g(model, i, j):
            return model.g[i, j] == model.DS[i, j] + model.delta_g[i, j]

        model.constr_actual_g = pyo.Constraint(model.indices, rule=constr_actual_g)

        ##All other constraints
        self.define_constraints(model)

        # Solve optimisation problem
        optimizer = pyo.SolverFactory('ipopt')
        optimizer.options['max_iter'] = 5000
        try:
            optimizer.solve(model)
            return(model)
        except ValueError as v:
            count_error = count_error + 1
            print(v, count_error)
            if count_error <= 5:
                actual_e_temp = {key: round(value + random.random() / 1000, 4) for key, value in actual_e.items()}
                return(self.controlled_ess(actual_e_temp, DS, actual_l, count_error))

    def estimate_soe(self, e, DS, forecast_l):
        offset_scheduling = [keys[1] for keys in forecast_l.keys()]
        for j, i in enumerate(offset_scheduling):
            DS_extract = {keys: values for keys, values in DS.items() if keys[1] == i}
            forecast_l_extract = {keys: values for keys, values in forecast_l.items() if keys[1] == i}
            model = self.controlled_ess(e, DS_extract, forecast_l_extract)
            self.results_to_dict(model, 'estimate_soe')
            e = {(keys[0], keys[1]): values for keys, values in model.e.extract_values().items() if keys[1] > i}
        return (e)

    def run_oneday_offline_deterministic(self, actual_e, forecast_l, DS_old, forecast_l_estimate_soe=None):
        time = [keys[1] for keys in forecast_l.keys()]
        # time = list(forecast_l.index.get_level_values(level=1))
        dates = np.unique(np.array([time[i].date() for i in range(len(time))]))

        # Estimate SoE
        if not forecast_l_estimate_soe is None:
            forecast_l_temp = {keys: values for keys, values in forecast_l_estimate_soe.items() if
                               keys[1].date() <= dates[0]}
        else:
            forecast_l_temp = {keys: values for keys, values in forecast_l.items() if keys[1].date() <= dates[0]}
        # forecast_l_temp = forecast_l[time[0]][forecast_l.index.get_level_values('time').day <= days[0]].to_dict()
        estimated_e = self.estimate_soe(actual_e, DS_old, forecast_l_temp)
        # print(estimated_e)

        # Compute DS
        forecast_l_temp = {keys: values for keys, values in forecast_l.items() if keys[1].date() >= dates[1]}
        # forecast_l_temp = forecast_l[time[0]][forecast_l.index.get_level_values('time').day >= days[1]].to_dict()
        results_compute_ds = self.compute_ds_deterministic(estimated_e, forecast_l_temp)
        self.results_to_dict(results_compute_ds, 'offline')


    def run_oneday_online(self, actual_e, actual_l, DS):
        time = [keys[1] for keys in actual_l.keys()]

        for hour in time:
            # Conrtolled ESS
            DS_temp = {(keys[0], keys[1]): values for keys, values in DS.items() if keys[1] == hour}
            actual_l_temp = {keys: values for keys, values in actual_l.items() if keys[1] ==  hour}
            #actual_l_temp = actual_l[time[0]][actual_l.index.get_level_values('time') == hour].to_dict()
            results_controlled_ess = self.controlled_ess(actual_e, DS_temp, actual_l_temp)
            self.results_to_dict(results_controlled_ess, 'online')
            actual_e = {(keys[0], keys[1]): values for keys, values in results_controlled_ess.e.extract_values().items()
                        if keys[1] > hour}

    def run_oneday_deterministic(self, actual_e_offline, actual_e_online, forecast_l, actual_l, DS_old,
                                 forecast_l_estimate_soe=None):
        time = [keys[1] for keys in forecast_l.keys()]
        # time = list(forecast_l.index.get_level_values(level=1))
        dates = np.unique(np.array([time[i].date() for i in range(len(time))]))

        self.run_oneday_offline_deterministic(actual_e_offline, forecast_l, DS_old, forecast_l_estimate_soe)

        DS = {(keys[0], keys[1]): values for keys, values in self.results_offline['g'].items() if
              keys[1].date() == dates[1]}
        self.run_oneday_online(actual_e_online, actual_l, DS)

    def run_firstday_deterministic(self, actual_e, forecast_l, actual_l):
        time = [keys[1] for keys in forecast_l.keys()]
        time0 = [keys[1] for keys in actual_l.keys()][0]
        #time = list(forecast_l.index.get_level_values(level=1))
        dates = np.unique(np.array([time[i].date() for i in range(len(time))]))

        e = {(keys[0], time0): values for keys, values in actual_e.items()}
        forecast_l_temp = {keys: values for keys, values in forecast_l.items() if keys[1].date() >= dates[1]}
        #forecast_l_temp = forecast_l[time[0]][forecast_l.index.get_level_values('time').day >= days[1]].to_dict()
        results_compute_ds = self.compute_ds_deterministic(e, forecast_l_temp)
        self.results_to_dict(results_compute_ds, 'offline')

        DS = {(keys[0], keys[1]): values for keys, values in self.results_offline['g'].items() if
              keys[1].day == dates[1].day}
        self.run_oneday_online(e, actual_l, DS)

    def run_deterministic_same_forecasts(self, list_actual_e, list_path_forecast_l, list_path_actual_l,
                                         start_computation=None, end_computation=None):
        self.forecast_path_list = list_path_forecast_l
        self.actual_path_list = list_path_actual_l
        # Read dataframe
        forecast = []
        actual = []
        for i, house in enumerate(self.houses):
            if not start_computation is None:
                forecast.append(pd.read_csv(list_path_forecast_l[i], parse_dates=["time"], index_col='time')[
                                start_computation:end_computation])
                actual.append(pd.read_csv(list_path_actual_l[i], parse_dates=["time"], index_col='time', skiprows=0)[
                              start_computation + pd.Timedelta(days=1):end_computation])
            else:
                forecast.append(pd.read_csv(list_path_forecast_l[i], parse_dates=["time"], index_col='time'))
                actual.append(pd.read_csv(list_path_actual_l[i], parse_dates=["time"], index_col='time', skiprows=0))
        timerange_forecast = [float(i) * pd.Timedelta(hours=self.delta) for i in forecast[0].columns]
        timerange_actual = [float(i) * pd.Timedelta(hours=self.delta) for i in actual[0].columns]

        is_first_day = True
        for t, time in enumerate(forecast[0].index):
            series_forecast = []
            series_actual = []
            # Convert dataframe to dict
            for i, house in enumerate(self.houses):
                temp_forecast = pd.Series(forecast[i].loc[time])
                temp_actual = pd.Series(actual[i].iloc[t])
                temp_forecast.index = [time + j for j in timerange_forecast]
                temp_actual.index = [actual[i].index[t] + j for j in timerange_actual]
                series_forecast.append(temp_forecast)
                series_actual.append(temp_actual)
            forecast_l = pd.concat(series_forecast, keys=self.houses)
            actual_l = pd.concat(series_actual, keys=self.houses)
            print(time)

            # Run OP for first day
            if is_first_day == True:
                actual_e = {(house, actual[0].index[0]): list_actual_e[h] for h, house in enumerate(self.houses)}
                self.run_firstday_deterministic(actual_e, forecast_l, actual_l)
                # save results
                self.dict_to_df(self.results_offline, self.results_offline_final)
                self.dict_to_df(self.results_online, self.results_online_final)
                is_first_day = False
            else:
                actual_e_offline = {(key[0], key[1]): value for key, value in self.results_online['e'].items() if
                                    key[1].day == time.day and key[1].hour == self.hour_start_computation}
                actual_e_online = {(key[0], key[1]): value for key, value in self.results_online['e'].items() if
                                   key[1].day == (time + pd.Timedelta(days=1)).day and key[
                                       1].hour == self.hour_start_schedule}
                DS_old = {(key[0], key[1]): value for key, value in self.results_offline['g'].items() if
                          key[1].day == time.day and key[1].hour >= self.hour_start_computation}
                self.clear_dict(self.results_online)
                self.clear_dict(self.results_offline)
                self.clear_dict(self.results_estimate_soe)
                self.run_oneday_deterministic(actual_e_offline, actual_e_online, forecast_l, actual_l, DS_old)
                # save results
                self.dict_to_df(self.results_estimate_soe, self.results_estimate_soe_final)
                self.dict_to_df(self.results_offline, self.results_offline_final)
                self.dict_to_df(self.results_online, self.results_online_final)

    def run_deterministic_different_forecasts(self, list_actual_e, list_path_forecast_l_estimate_soe,
                                              list_path_forecast_l_offline, list_path_actual_l, start_computation=None,
                                              end_computation=None):
        self.forecast_offline_path_list = list_path_forecast_l_offline
        self.actual_path_list = list_path_actual_l
        # Read dataframe
        forecast_estimate_soe = []
        forecast_offline = []
        actual = []
        for i, house in enumerate(self.houses):
            if not start_computation is None:
                forecast_estimate_soe.append(
                    pd.read_csv(list_path_forecast_l_estimate_soe[i], parse_dates=["time"], index_col='time')[
                    start_computation:end_computation])
                forecast_offline.append(
                    pd.read_csv(list_path_forecast_l_offline[i], parse_dates=["time"], index_col='time')[
                    start_computation:end_computation])
                actual.append(pd.read_csv(list_path_actual_l[i], parse_dates=["time"], index_col='time', skiprows=0)[
                              start_computation + pd.Timedelta(days=1):end_computation])
            else:
                forecast_offline.append(
                    pd.read_csv(list_path_forecast_l_estimate_soe[i], parse_dates=["time"], index_col='time'))
                forecast_offline.append(
                    pd.read_csv(list_path_forecast_l_offline[i], parse_dates=["time"], index_col='time'))
                actual.append(pd.read_csv(list_path_actual_l[i], parse_dates=["time"], index_col='time', skiprows=0))
        timerange_forecast_estimate_soe = [float(i) * pd.Timedelta(hours=self.delta) for i in
                                           forecast_estimate_soe[0].columns]
        timerange_forecast_offline = [float(i) * pd.Timedelta(hours=self.delta) for i in forecast_offline[0].columns]
        timerange_actual = [float(i) * pd.Timedelta(hours=self.delta) for i in actual[0].columns]

        is_first_day = True
        for t, time in enumerate(forecast_offline[0].index):
            series_forecast_estimate_soe = []
            series_forecast_offline = []
            series_actual = []
            # Convert dataframe to dict
            for i, house in enumerate(self.houses):
                temp_forecast_estimate_soe = pd.Series(forecast_estimate_soe[i].loc[time])
                temp_forecast_offline = pd.Series(forecast_offline[i].loc[time])
                temp_actual = pd.Series(actual[i].iloc[t])
                temp_forecast_estimate_soe.index = [time + j for j in timerange_forecast_estimate_soe]
                temp_forecast_offline.index = [time + j for j in timerange_forecast_offline]
                temp_actual.index = [actual[i].index[t] + j for j in timerange_actual]
                series_forecast_estimate_soe.append(temp_forecast_estimate_soe)
                series_forecast_offline.append(temp_forecast_offline)
                series_actual.append(temp_actual)
            forecast_l_estimate_soe = pd.concat(series_forecast_estimate_soe, keys=self.houses)
            forecast_l_offline = pd.concat(series_forecast_offline, keys=self.houses)
            actual_l = pd.concat(series_actual, keys=self.houses)
            print(time)

            # Run OP for first day
            if is_first_day == True:
                actual_e = {(house, actual[0].index[0]): list_actual_e[h] for h, house in enumerate(self.houses)}
                self.run_firstday_deterministic(actual_e, forecast_l_offline, actual_l)
                # save results
                self.dict_to_df(self.results_offline, self.results_offline_final)
                self.dict_to_df(self.results_online, self.results_online_final)
                is_first_day = False
            else:
                actual_e_offline = {(key[0], key[1]): value for key, value in self.results_online['e'].items() if
                                    key[1].day == time.day and key[1].hour == self.hour_start_computation}
                actual_e_online = {(key[0], key[1]): value for key, value in self.results_online['e'].items() if
                                   key[1].day == (time + pd.Timedelta(days=1)).day and key[
                                       1].hour == self.hour_start_schedule}
                DS_old = {(key[0], key[1]): value for key, value in self.results_offline['g'].items() if
                          key[1].day == time.day and key[1].hour >= self.hour_start_computation}
                self.clear_dict(self.results_online)
                self.clear_dict(self.results_offline)
                self.clear_dict(self.results_estimate_soe)
                self.run_oneday_deterministic(actual_e_offline, actual_e_online, forecast_l_offline, actual_l, DS_old,
                                              forecast_l_estimate_soe)
                # save results
                self.dict_to_df(self.results_estimate_soe, self.results_estimate_soe_final)
                self.dict_to_df(self.results_offline, self.results_offline_final)
                self.dict_to_df(self.results_online, self.results_online_final)

    def run_deterministic_fixed_soe(self, list_e, list_path_forecast_l, list_path_actual_l, start_computation=None,
                                    end_computation=None):
        self.forecast_path_list = list_path_forecast_l
        self.actual_path_list = list_path_actual_l
        # Read dataframe
        forecast = []
        actual = []
        for i, house in enumerate(self.houses):
            if not start_computation is None:
                forecast.append(pd.read_csv(list_path_forecast_l[i], parse_dates=["time"], index_col='time')[
                                start_computation:end_computation])
                actual.append(pd.read_csv(list_path_actual_l[i], parse_dates=["time"], index_col='time', skiprows=0)[
                              start_computation + pd.Timedelta(days=1):end_computation])
            else:
                forecast.append(pd.read_csv(list_path_forecast_l[i], parse_dates=["time"], index_col='time'))
                actual.append(pd.read_csv(list_path_actual_l[i], parse_dates=["time"], index_col='time', skiprows=0))
        timerange_forecast = [float(i) * pd.Timedelta(hours=self.delta) for i in forecast[0].columns]
        timerange_actual = [float(i) * pd.Timedelta(hours=self.delta) for i in actual[0].columns]

        for t, time in enumerate(forecast[0].index):
            series_forecast = []
            series_actual = []
            # Convert dataframe to dict
            for i, house in enumerate(self.houses):
                temp_forecast = pd.Series(forecast[i].loc[time])
                temp_actual = pd.Series(actual[i].iloc[t])
                temp_forecast.index = [time + j for j in timerange_forecast]
                temp_actual.index = [actual[i].index[t] + j for j in timerange_actual]
                series_forecast.append(temp_forecast)
                series_actual.append(temp_actual)
            forecast_l = pd.concat(series_forecast, keys=self.houses)
            actual_l = pd.concat(series_actual, keys=self.houses)
            print(time)

            # Run OP
            actual_e = {(house, actual[0].index[t]): list_e[h].loc[actual[0].index[t]].squeeze() for h, house in
                        enumerate(self.houses)}
            self.clear_dict(self.results_online)
            self.clear_dict(self.results_offline)
            self.run_firstday_deterministic(actual_e, forecast_l, actual_l)
            # save results
            self.dict_to_df(self.results_offline, self.results_offline_final)
            self.dict_to_df(self.results_online, self.results_online_final)

    def define_constraints(self, model):
        ##g = p + l
        def constr_power_flow(model, i, j):
            return model.g[i, j] == model.p[i, j] + model.l[i, j]

        model.constr_power_flow = pyo.Constraint(model.indices, rule=constr_power_flow)

        ##p = p+ + p-
        def constr_split_p(model, i, j):
            return model.p[i, j] == model.p_plus[i, j] + model.p_minus[i, j]

        model.constr_split_p = pyo.Constraint(model.indices, rule=constr_split_p)

        ##p+ * p- <= relaxation_p
        def constr_relax_p(model, i, j):
            return (-self.relaxation_p[i], model.p_plus[i, j] * model.p_minus[i, j], 0)

        model.constr_relax_p = pyo.Constraint(model.indices, rule=constr_relax_p)

        def constr_evolution_e(model, i, j):
            if j == model.indices_time.first():
                return model.e[i, j] == model.e0[i, j]
            else:
                return model.e[i, j] == model.e[i, model.indices_time_e.prev(j)] + self.delta * (
                        model.p[i, model.indices_time_e.prev(j)] - self.mu[i] * model.p_plus[
                    i, model.indices_time_e.prev(j)] + \
                        self.mu[i] * model.p_minus[i, model.indices_time_e.prev(j)])

        model.constr_evolution_e = pyo.Constraint(model.indices_houses, model.indices_time_e, rule=constr_evolution_e)

        ##-g_max <= sum(g) <= g_max
        def constr_g_max(model, j):
            return (-self.g_max, sum(model.g[i, j] for i in model.indices_houses), self.g_max)

        model.constr_g_max = pyo.Constraint(model.indices_time, rule=constr_g_max)

    def define_variables(self, model):
        model.g = pyo.Var(model.indices)
        model.p = pyo.Var(model.indices)

        def bounds_p_plus(model, i, j):
            return (0, self.p_max[i])

        model.p_plus = pyo.Var(model.indices, bounds=bounds_p_plus)

        def bounds_p_min(model, i, j):
            return (self.p_min[i], 0)

        model.p_minus = pyo.Var(model.indices, bounds=bounds_p_min)

        def bounds_e(model, i, j):
            return (self.e_min[i], self.e_max[i])

        model.e = pyo.Var(model.indices_houses, model.indices_time_e, bounds=bounds_e)

    def results_to_dict(self, model, string_level):
        vars = model.component_map(ctype=pyo.Var)
        results_dict = getattr(self, 'results_' + string_level)
        #results_dict_temp = copy.deepcopy(results_dict)
        for key, v in vars.items():
            for u, vv in v.items():
                results_dict[key][u] = vv.value
        setattr(self, 'results_' + string_level, results_dict)

    def dict_to_df(self, dicti, dicti_final):
        #time = np.arange(len(np.unique(np.array([keys[1] for keys in variables.keys()]))))
        for vari in dicti:
            dict_temp = dicti[vari]
            index = list(dict_temp.keys())[0][1]
            for house in self.houses:
                temp = pd.DataFrame([value for key, value in dict_temp.items() if key[0] == house], columns=[index]).transpose()
                dicti_final[vari][house] = pd.concat([dicti_final[vari][house], temp])

    def clear_dict(self, dicti):
        for key in dicti:
            dicti[key] = {}
    
    def calculate_ds_costs_daily(self, str_house):
        self.ds_costs_daily = reduce(lambda a, b: a.add(b, fill_value=0), [
            self.results_offline_final['g_plus'][str_house].iloc[:, :24].multiply(self.c_lin_plus).sum(axis=1),
            (self.results_offline_final['g_plus'][str_house] ** 2).iloc[:, :24].multiply(self.c_quadr_plus).sum(axis=1),
            self.results_offline_final['g_minus'][str_house].iloc[:, :24].multiply(self.c_lin_minus).sum(axis=1),
            (self.results_offline_final['g_minus'][str_house] ** 2).iloc[:, :24].multiply(self.c_quadr_minus).sum(
                axis=1)])

    def calculate_imbalance_costs_daily(self, str_house):
        self.imbalance_costs_daily = (self.results_offline_final['g'][str_house].iloc[:, :24].subtract(
            self.results_online_final['g'][str_house]) ** 2).multiply(self.c_quadr_plus).sum(axis=1).add(
            (self.results_offline_final['g'][str_house].iloc[:, :24].subtract(
                self.results_online_final['g'][str_house])).abs().multiply(self.c_lin_plus).sum(axis=1), fill_value=0)
