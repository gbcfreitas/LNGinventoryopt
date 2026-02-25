import pulp
import numpy as np
import logging
import pandas as pd
import json


class otimizador:
    """Função para definição do otimizador ee xtração de resultados"""

    def __init__(self, nome, arquivo_json, arquivo_excel, options) -> None:

        self.nome = nome
        self.run = options["RUN"]
        self.timeLimit = options["TIMELIMIT"]
        self.options = options
        logging.info(f"Carregando parametros para o cenario {nome}.")
        self.entradas = self.carregar_parametros(nome, arquivo_json, arquivo_excel)
        logging.info(
            f"Parametros carregados para o cenario {nome}. Iniciando otimizacao."
        )
        
        
        # self.model = self.solve_model()

        self.resultado_modelo = {
            "status": "",
            "objective_value": 0,
            "solution_time": 0,
            "x": [],
            "s": [],
            "b": [],
            "y_bog": [],
            "cc_acc": [],
            "y_top": [],
            "ca_acc": [],
            "y_price": [],
            "v": []
        }

    @staticmethod
    def results_dict_to_list(
        variables_dict, var: str, index1, index2=None, index3=None
    ):
        if index1 and index2 and index3:
            # Uso de compreensão de listas para atribuir os valores
            var_values = [
                [
                    [variables_dict[f"{var}_{j}_{m}_{t}"].value() for t in index3]
                    for m in index2
                ]
                for j in index1
            ]
        elif index1 and index2:
            var_values = [
                [variables_dict[f"{var}_{j}_{m}"].value() for m in index2]
                for j in index1
            ]
        else:
            var_values = [variables_dict[f"{var}_{j}"].value() for j in index1]
        return var_values

    def carregar_parametros(self, nome, arquivo_json, arquivo_excel):
        """
        Carrega parâmetros de um arquivo JSON e converte para o formato de tuplas.

        Args:
            arquivo_json (str): Caminho para o arquivo JSON

        Returns:
            tuple: (params_modelo_convertido, params_cenario, params_operacao)
        """

        with open(arquivo_json, "r", encoding="utf-8") as f:
            dados = json.load(f)[nome]

        params_por_contrato = dados["parametros_por_contrato"]
        indices = dados["indices"]
        conjuntos = dados["conjuntos"]
        parametros_modelo = dados["parametros_modelo"]
        
        if self.options["LOCK_POLICY"]:
            variaveis_fixadas = dados["variaveis_fixadas"]

        self.entradas = {}
        self.entradas.update({key: range(value) for key, value in indices.items()})
        if self.options["LOCK_POLICY"]:
            self.entradas.update({key: value for key, value in variaveis_fixadas.items()})
        self.entradas.update(conjuntos)
        self.entradas.update(parametros_modelo)
        
        if self.options["LOCK_POLICY"]:
            # if self.options["SOLVER"]
            self.entradas["y"] = variaveis_fixadas["y"]



        for j, contrato in params_por_contrato.items():
            for param, valor in contrato.items():
                if param in ["V_top_up", "V_top_lo", "V_price_up", "V_price_lo"]:
                    self.entradas[param] = {}
                    for k, v in enumerate(valor):
                        self.entradas[param][(int(j), k)] = v
                else:
                    self.entradas[param] = {int(j): valor}

        # Aplicar conversão nos dados do Excel

        params_tuplas = otimizador.converter_excel_para_tuplas(
            arquivo_excel, aba_nome=nome
        )
        self.entradas.update(params_tuplas)

        CI = {}
        for j in range(len(self.entradas["J"]) - 1):
            int_ = []
            ini = 0
            fim = 0
            i_1 = 0
            T = self.entradas["T"]
            for t in T:
                i_0 = self.entradas["V_adp"][(j, t)]
                if i_0 == 0 and i_1 == 0:
                    pass
                elif i_0 != 0 and i_1 == 0:
                    ini = t
                elif i_0 == 0 and i_1 != 0:
                    fim = t - 1
                    int_.append([ini, fim])
                else:
                    pass
                i_1 = i_0

            CI[j] = np.array(int_)
        self.entradas.update({"CI": CI})

        return self.entradas

    @staticmethod
    def converter_excel_para_tuplas(arquivo_excel, aba_nome):
        """
        Converte parâmetros do Excel para formato de tuplas usado no modelo.
        Para cada coluna D_i, W_j, etc., cria dicionário com chaves (i,t) ou (j,t).
        """

        try:
            df = pd.read_excel(arquivo_excel, sheet_name=aba_nome, index_col=0)

            parametros = {}
            for coluna in df.columns:
                parametros[coluna] = {}
                for index, valor in df[coluna].items():
                    if pd.notna(valor):
                        parametros[coluna][index] = valor

        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_excel}")
        except ValueError as e:
            if "Worksheet named" in str(e):
                with pd.ExcelFile(arquivo_excel) as xls:
                    available_sheets = xls.sheet_names
                raise ValueError(
                    f"Aba '{aba_nome}' não encontrada. Abas disponíveis: {available_sheets}"
                )
            else:
                raise ValueError(f"Erro ao ler arquivo: {e}")

        params_convertidos = {}

        # Separar por tipo de parâmetro
        for coluna in df.columns:
            if coluna.startswith("V_d_"):
                # Parâmetro V_d indexado por (i,t)
                i = int(coluna.split("_")[2])  # Extrair índice i do nome da coluna
                if "V_d" not in params_convertidos:
                    params_convertidos["V_d"] = {}

                for t, valor in parametros[coluna].items():
                    params_convertidos["V_d"][(i, t)] = valor

            elif coluna.startswith("V_adp_"):
                # Parâmetro V_adp indexado por (j,t)
                j = int(coluna.split("_")[2])  # Extrair índice j do nome da coluna
                if "V_adp" not in params_convertidos:
                    params_convertidos["V_adp"] = {}

                for t, valor in parametros[coluna].items():
                    params_convertidos["V_adp"][(j, t)] = valor

            elif coluna.startswith("N_dem_"):
                # Parâmetro N_dem indexado por (j,t)
                j = int(coluna.split("_")[2])
                if "N_dem" not in params_convertidos:
                    params_convertidos["N_dem"] = {}

                for t, valor in parametros[coluna].items():
                    params_convertidos["N_dem"][(j, t)] = valor


            elif coluna.startswith("P_spa_top_"):
                # Parâmetro "P_spa_top" indexado por (j,m,t)
                j = int(coluna.split("_")[1+2])
                m = int(coluna.split("_")[2+2])
                if "P_spa_top" not in params_convertidos:
                    params_convertidos["P_spa_top"] = {}

                for t, valor in parametros[coluna].items():
                    params_convertidos["P_spa_top"][(j, m, t)] = valor

            elif coluna.startswith("P_spa_price_"):
                # Parâmetro P_spa_price indexado por (j,m,t)
                j = int(coluna.split("_")[1+2])
                c = int(coluna.split("_")[2+2])
                if "P_spa_price" not in params_convertidos:
                    params_convertidos["P_spa_price"] = {}

                for t, valor in parametros[coluna].items():
                    params_convertidos["P_spa_price"][(j, c, t)] = valor

            elif coluna == "Idle":
                # Parâmetro Idle indexado por t
                if "Idle" not in params_convertidos:
                    params_convertidos["Idle"] = {}

                for t, valor in parametros[coluna].items():
                    params_convertidos["Idle"][t] = valor

            elif coluna == "P_spot":
                # Parâmetro P_spot indexado por t
                if "P_spot" not in params_convertidos:
                    params_convertidos["P_spot"] = {}

                for t, valor in parametros[coluna].items():
                    params_convertidos["P_spot"][t] = valor

        return params_convertidos

    def solve_model(self) -> pulp.LpProblem:
        """
        Escrito com o framework PulP utilizando a API com o CBC. A descrição da classe pulp.PULP_CBC_CMD é encontrada em https://coin-or.github.io/pulp/technical/solvers.html#pulp.apis.PULP_CBC_CMD.name

        Args:
            parametros_modelo (dict): _description_
            parametros_cenario (dict): _description_
            number_index (_type_): _description_
            dolar (_type_): _description_

        Returns:
            pulp.LpProblem: _description_
        """

        # LEITURA DE DADOS DE ENTRADA

        # INDICES E CONJUNTOS
        T, I, J, M, C, N = (
            self.entradas["T"],
            self.entradas["I"],
            self.entradas["J"],
            self.entradas["M"],
            self.entradas["C"],
            self.entradas["N"],
        )
        J1 = range(len(self.entradas["J"]) - 1)
        self.entradas["J1"] = J1

        # PARAMETROS DO MODELO
        (
            K,
            S_0,
            S_up,
            S_idle_lo,
            S_prod_lo,
            V_spot_lo,
            V_spot_up,
            P_dem,
            V_bog_up,
            V_bog_lo,
            F_bog,
            V_bog_prod,
        ) = (
            self.entradas["K"],
            self.entradas["S_0"],
            self.entradas["S_up"],
            self.entradas["S_idle_lo"],
            self.entradas["S_prod_lo"],
            self.entradas["V_spot_lo"],
            self.entradas["V_spot_up"],
            self.entradas["P_dem"],
            self.entradas["V_bog_up"],
            self.entradas["V_bog_lo"],
            self.entradas["F_bog"],
            self.entradas["V_bog_prod"],

        )


        # PARAMETROS POR CONTRATO
        CI, V_total, V_cc_0, V_ca_0, V_top_up, V_top_lo, V_price_up, V_price_lo = (
            self.entradas["CI"],
            self.entradas["V_total"],
            self.entradas["V_cc_0"],
            self.entradas["V_ca_0"],
            self.entradas["V_top_up"],
            self.entradas["V_top_lo"],
            self.entradas["V_price_up"],
            self.entradas["V_price_lo"],
        )
        # for j in CI:
        #     CI[j] = np.array(CI[j])

        # PARAMETROS INDEXADOS NO TEMPO
        V_d, V_adp, Idle, N_dem, P_spot, P_spa_top, P_spa_price = (
            self.entradas["V_d"],
            self.entradas["V_adp"],
            self.entradas["Idle"],
            self.entradas["N_dem"],
            self.entradas["P_spot"],
            self.entradas["P_spa_top"],
            self.entradas["P_spa_price"],
        )

        # if self.options["CAPCOST"]:
        R = self.entradas["R"]
        P_cap = self.entradas["P_cap"]

        logging.info("Iniciando definição do modelo de otimização.")

        ## VARIAVEIS
        if self.options["LOCK_POLICY"] == False:
            y = pulp.LpVariable.dicts("y", (J, T), cat="Binary")
        else:
            # y = self.entradas["y"]
            y = pulp.LpVariable.dicts("y", (J, T), cat="Binary")
            for j in J1:
                for t in T:
                    y[j][t].setInitialValue(self.entradas["y"][j][t])
                    y[j][t].fixValue()

        x = pulp.LpVariable.dicts("x", T, lowBound=0, upBound=V_spot_up)
        s = pulp.LpVariable.dicts("s", T, lowBound=0, upBound=S_up)
        b = pulp.LpVariable.dicts("b", T, lowBound=0)
        y_bog = pulp.LpVariable.dicts("y_bog", (T, N), cat="Binary")
        cc_acc = pulp.LpVariable.dicts("cc_acc", (J1, T), lowBound=0)
        y_top = pulp.LpVariable.dicts("y_top", (J1, M, T), cat="Binary")
        ca_acc = pulp.LpVariable.dicts("ca_acc", (J1, T), lowBound=0)
        y_price = pulp.LpVariable.dicts("y_price", (J1, C, T), cat="Binary")
        v = pulp.LpVariable.dicts("v", T)



        logging.info("Variaveis definidas com sucesso.")
        # # DEFINICAO DO PROBLEMA
        # if self.options["SOLVER"] == "PULP_CBC_CMD":
        #     solver = pulp.PULP_CBC_CMD(
        #         timeLimit=self.timeLimit, logPath=f"saidas/cbc_run{self.run}.log"
        #     )
        # elif self.options["SOLVER"] == "SCIP_CMD":
        #     solver = pulp.SCIP_CMD(
        #         timeLimit=self.timeLimit, logPath=f"saidas/scip_run{self.run}.log"
        #     )
        # elif self.options["SOLVER"] == "GLPK_CMD":
        #     solver = pulp.GLPK_CMD(timeLimit=self.timeLimit, keepFiles=True)
        # elif self.options["SOLVER"] == "CPLEX_CMD":
        solver = pulp.CPLEX_CMD(
            timeLimit=self.timeLimit, logPath=f"saidas/cplex_run{self.run}.log"
        )

        logging.info("Solver definido com sucesso.")
        # solver = pulp.PULP_CBC_CMD(threads = 6, timeLimit= 60)
        prob = pulp.LpProblem("Simulacao_cenarios", pulp.LpMinimize)

        

        # FUNCAO OBJETIVO
        def objetive_func():
            total = 0
            for t in T:
                total += pulp.lpSum(y[j][t] for j in J) * K
                total += x[t] * P_spot[(t)]
                total += pulp.lpSum(
                    P_spa_price[(j, c, t)] * V_adp[(j, t)] * y_price[j][c][t] for j in J1 for c in C
                )
                total += pulp.lpSum(y[j][t] * P_dem[j] * N_dem[(j, t)] for j in J1)
                total += pulp.lpSum(
                    y_top[j][m][t] * V_adp[(j, t)] * P_spa_top[(j, m, t)] for j in J1 for m in M
                )

                if self.options["CAPCOST"]:
                    total += v[t] * R

            return total

        prob += objetive_func(), "Custo Total"
        
        logging.info("Funcao objetivo definida com sucesso.")

        def constr_store_balance_no_bog(prob, T, s, x, y, V_d, V_adp, I, J1, S_0):
            """
            Balanço de massa sem considerar bog

            Referência Dissertação
            Seção: 2.2.2 Inventory Physical and Operational Constraints
            Equação: Equation 1
            """
            for t in T:
                if t == 0:
                    prob += (
                        s[t]
                        == S_0
                        + x[t]
                        + pulp.lpSum(V_adp[(j, t)] * y[j][t] for j in J1)
                        - pulp.lpSum(V_d[(i, t)] for i in I),
                        f"Store_Balance_Bog_Cte_Constraint_{t}",
                    )

                else:
                    prob += (
                        s[t]
                        == s[t - 1]
                        + x[t]
                        + pulp.lpSum(V_adp[(j, t)] * y[j][t] for j in J1)
                        - pulp.lpSum(V_d[(i, t)] for i in I),
                        f"Store_Balance_Bog_Cte_Constraint_{t}",
                    )

        def constr_store_capacity_lower_idle(prob, T, S_idle_lo, Idle, s):
            """
            Limite de inventário mínimo para períodos Idle

            Referência Dissertação
            Seção: 2.2.2 Inventory Physical and Operational Constraints
            Equação: Equation 2
            """
            for t in T:
                prob += (
                    s[t] >= S_idle_lo * Idle[t],
                    f"Store_Capacity_L_Idle_Constraint_{t}",
                )

        def constr_store_capacity_lower_prod(
            prob, T, S_prod_lo, s, Idle
        ):
            """
            Limite de inventário mínimo para períodos PROD

            Referência Dissertação
            Seção: 2.2.2 Inventory Physical and Operational Constraints
            Equação: Equation 3
            """
            for t in T:
                prob += (
                    s[t] >= S_prod_lo * (1 - Idle[t]),
                    f"Store_Capacity_L_Prod_Constraint_{t}",
                )
                
        logging.info("Limite de inventario minimo para períodos PROD definido com sucesso.")

        def constr_gap_between_arrival(prob, T, J, y):
            """
            Limit operational constraint for ship arrival intervals.
            Ensures no more than one cargo arrives within 3 consecutive periods.
            """
            for t in range(len(T)):
                if t == 0:
                    # At t=0, only check current period
                    prob += (
                        sum(y[j][t] for j in range(len(J))) <= 1,
                        f"Interval_Constraint_{t}",
                    )
                elif t == 1:
                    # At t=1, check current and previous period
                    prob += (
                        sum(y[j][t] + y[j][t - 1] for j in range(len(J))) <= 1,
                        f"Interval_Constraint_{t}",
                    )
                else:
                    # At t>=2, check 3-period window
                    prob += (
                        sum(y[j][t] + y[j][t - 1] + y[j][t - 2] for j in range(len(J)))
                        <= 1,
                        f"Interval_Constraint_{t}",
                    )
        logging.info("Limite de intervalo entre chegadas definido com sucesso.")
        
        
        def constr_interval_limit_one_load(prob, J1, CI, y):
            """
            Garante que, para cada contrato j in J1 e para cada conjunto CI_j
            que contém os conjuntos de períodos de tempo em que cada carga pode ser chamada,
            apenas 1 período possa ser selecionado
            e exclui o período "fim", em coerência com o uso de range(interval[0], interval[1]).

            Referência Dissertação
            Seção: 2.2.3 Modeling of Procurement Contracts
            Equação: Equation 7
            """
            # constr_demurrage_interval = []
            for j in J1:
                for interval in CI[j]:
                    prob += (
                        pulp.lpSum(y[j][t] for t in range(interval[0], interval[1])) <= 1,
                        f"Demurrage_Interval_Constraint_{j}_{interval[0]}_{interval[1]}",
                    )
                    # constr_demurrage_interval.append(
                    #     prob.addConstraint(
                    #         pulp.lpSum(y[j][t] for t in range(interval[0], interval[1]))
                    #         <= 1,
                    #         f"Demurrage_Interval_Constraint_{j}_{interval[0]}_{interval[1]}",
                    #     )
                    # )
                    
            logging.info("Limite de compra por intervalo definido com sucesso.")

        def constr_lng_cost(prob, T, J1, C, V_price_lo, V_price_up, y_price, ca_acc, V_adp, V_ca_0):
            """
            Define as restrições de custo de GNL (LNG cost) associadas aos contratos
            em função do volume acumulado entregue, utilizando uma representação em
            faixas (piecewise linear) do custo marginal.

            Referência Dissertação
            Seção: 2.2.3 "Modeling of Procurement Contracts"
            Equação: Equation 8-11
            """
            constr_lgc_faixa = []
            for t in T:
                for j in J1:
                    for c in C:
                        constr_lgc_faixa.append(
                            V_price_lo[(j, c)] * y_price[j][c][t] - ca_acc[j][t] <= 0
                        )
                        constr_lgc_faixa.append(
                            V_price_up[(j, c)] * y_price[j][c][t]
                            + V_total[j] * (1 - y_price[j][c][t])
                            - ca_acc[j][t]
                            >= 0
                        )
                    if t == 0:
                        constr_lgc_faixa.append(
                            ca_acc[j][t]
                            == pulp.lpSum([y_price[j][c][t] for c in C]) * V_adp[(j, t)]
                            + V_ca_0[j]
                        )
                    else:
                        constr_lgc_faixa.append(
                            ca_acc[j][t]
                            == ca_acc[j][t - 1]
                            + pulp.lpSum(y_price[j][c][t] for c in C) * V_adp[(j, t)]
                        )

                    constr_lgc_faixa.append(
                        pulp.lpSum(y_price[j][c][t] for c in C) == y[j][t]
                    )

            for idx, constraint in enumerate(constr_lgc_faixa):
                prob += constraint, f"Lgc_Faixa_Constraint_{idx}"

        def constr_top(prob, T, J1, M, V_total, cc_acc, V_adp, V_cc_0, y_top):
            """
            Define as restrições de custo de Take or Pay associadas aos contratos
            em função do volume acumulado canceladio, utilizando uma representação em
            faixas (piecewise linear) do custo marginal.

            Referência Dissertação
            Seção: 2.2.3 "Modeling of Procurement Contracts"
            Equação: Equation 12-16
            """
            constr_topc_faixa = []
            for t in T:
                for j in J1:
                    for m in M:
                        constr_topc_faixa.append(
                            V_top_lo[(j, m)] * y_top[j][m][t] - cc_acc[j][t] <= 0
                        )
                        constr_topc_faixa.append(
                            V_top_up[(j, m)] * y_top[j][m][t]
                            + V_total[j] * (1 - y_top[j][m][t])
                            - cc_acc[j][t]
                            >= 0
                        )
                    if t == 0:
                        constr_topc_faixa.append(
                            cc_acc[j][t]
                            == pulp.lpSum(y_top[j][m][t] for m in M) * V_adp[(j, t)]
                            + V_cc_0[j]
                        )
                    else:
                        constr_topc_faixa.append(
                            cc_acc[j][t]
                            == cc_acc[j][t - 1]
                            + pulp.lpSum([y_top[j][m][t] for m in M]) * V_adp[(j, t)]
                        )
            for idx, constraint in enumerate(constr_topc_faixa):
                prob += constraint, f"Topc_Faixa_Constraint_{idx}"

            for j in J1:
                for interval in CI[j]:
                    for t in range(interval[0], interval[1]):
                        if t == interval[0]:
                            prob += (
                                pulp.lpSum(y_top[j][m][t] for m in M)
                                == (
                                    1
                                    - pulp.lpSum(
                                        y[j][t] for t in range(interval[0], interval[1])
                                    )
                                ),
                                f"Topc_Constraint_1_{j}_{t}",
                            )
                            prob += (
                                pulp.lpSum(
                                    y_top[j][m][t]
                                    for t in range(interval[0], interval[1])
                                    for m in M
                                )
                                <= (
                                    1
                                    - pulp.lpSum(
                                        y[j][t] for t in range(interval[0], interval[1])
                                    )
                                ),
                                f"Topc_Constraint_2_{j}_{t}",
                            )

        def constr_spot_size_high(prob, T, V_spot_up):
            """
            Define o limite máximo para a compra spot, que é dada pela variável x[t], e deve ser limitada por V_spot_up, que representa a capacidade máxima de compra spot em um período.

            Referência Dissertação
            Seção: 2.2.3 "Modeling of Procurement Contracts"
            Equação: Equation 17-18
            """
            for t in T:
                prob += x[t] <= V_spot_up * y[len(J) - 1][t], f"Spot_Constraint_up_{t}"
                prob += x[t] >= V_spot_lo * y[len(J) - 1][t], f"Spot_Constraint_lo_{t}"



        
        #         # Tipo de modelo

        def constr_bog_level(prob, T, Idle, s, y_bog, V_bog_lo, V_bog_up, S_up, b, F_bog, V_bog_prod):
            """
            Define a relação entre volume atual (s) e BOG gerado (F_bog) a partir dos intervalos de volume definidos por V_bog_lo e V_bog_up
            
            Referência Dissertação
            Seção: 2.2.4 "Modeling of Boil-Off Gas (BOG) Losses"
            Equação: Equation 20-23
            """
            for t in T:
                if Idle[t] == 1:
                    for n in N:
                        prob += V_bog_lo[n] * y_bog[t][n] - s[t] <= 0, f"Boglevel_Lower_{t}_{n}"
                        prob += (
                            V_bog_up[n] * y_bog[t][n] + S_up * (1 - y_bog[t][n]) - s[t] >= 0,
                            f"Boglevel_Upper_{t}_{n}",
                        )

                    prob += sum(y_bog[t][n] for n in N) == 1, f"Boglevel_Sum_{t}"
                    prob += b[t] == sum(F_bog[n] * y_bog[t][n] for n in N), f"Boglevel_b_{t}"

                else:
                    prob += b[t] == V_bog_prod, f"Boglevel_b_fixed_{t}"
                    prob += sum(y_bog[t][n] for n in N) == 0, f"Boglevel_by_fixed_{t}"

        
        def constr_store_balance_bog(
            prob, T, s, x, y, V_d, V_adp, I, J1, S_0, b, Idle, V_bog_prod
        ):
            """
            Substitui a equação de balanço de massa para considerar as perdas por BOG, representadas pela variável b[t], que é subtraída do lado direito da equação, reduzindo o volume disponível no período t.
            
            Referência Dissertação
            Seção: 2.2.5 "Modeling of Boil-Off Gas (BOG) Losses"
            Equação: Equation 24
            """
            for t in T:
                if t == 0:
                    prob += (
                        s[t]
                        == S_0
                        + x[t]
                        + pulp.lpSum(V_adp[(j, t)] * y[j][t] for j in J1)
                        - pulp.lpSum(V_d[(i, t)] for i in I)
                        - b[t] ,
                        f"Store_Balance_Bog_Cte_Constraint_{t}",
                    )

                else:
                    prob += (
                        s[t]
                        == s[t - 1]
                        + x[t]
                        + pulp.lpSum(V_adp[(j, t)] * y[j][t] for j in J1)
                        - pulp.lpSum(V_d[(i, t)] for i in I)
                        - b[t],
                        f"Store_Balance_Bog_Cte_Constraint_{t}",
                    )

        

        # Custo Financeiro de Capital
        def constr_capital_cost(prob, T, v, R, P_cap, s):
            """
                Define o custo financeiro de capital associado ao investimento necessário para cobrir a capacidade de armazenamento, representado pela variável v[t], que é calculada com base no volume armazenado s[t] e na capacidade de armazenamento P_cap, considerando a taxa de retorno R.
            
            Referência Dissertação
            Seção: 2.2.6 "Inventory Holding and Capital Cost Representation"
            Equação: Equation 25
            """
            for t in T:
                if t == 0:
                    prob += v[t] == s[t] * P_cap + x[t] * P_cap + pulp.lpSum(
                        V_adp[(j, t)] * y[j][t] * P_cap for j in J1
                    ) - pulp.lpSum(V_d[(i, t)] * P_cap for i in I)

                else:
                    prob += (
                        v[t]
                        == v[t - 1]
                        + v[t - 1] * R
                        + x[t] * P_cap
                        + pulp.lpSum(V_adp[(j, t)] * y[j][t] * P_cap for j in J1)
                        - pulp.lpSum(V_d[(i, t)] * P_cap for i in I),
                        f"Capital_Cost_Constraint_{t}",
                    )


        # Montar problema
        # Restrições BOG
        # if self.options["BOG_VAR"] == False:
        #     constr_store_balance_no_bog(prob, T, s, x, y, V_d, V_adp, I, J1, S_0)

        if self.options["BOG_VAR"]:
            constr_store_balance_bog(
                prob, T, s, x, y, V_d, V_adp, I, J1, S_0, b, Idle, V_bog_prod
            )
            constr_bog_level(prob, T, Idle, s, y_bog, V_bog_lo, V_bog_up, S_up, b, F_bog, V_bog_prod)
        else:
            constr_store_balance_no_bog(prob, T, s, x, y, V_d, V_adp, I, J1, S_0)

        constr_store_capacity_lower_idle(prob, T, S_idle_lo, Idle, s)
        constr_store_capacity_lower_prod(prob, T, S_prod_lo, s, Idle)
        constr_gap_between_arrival(prob, T, J, y)
        constr_interval_limit_one_load(prob, J1, CI, y)
        constr_lng_cost(prob, T, J1, C, V_price_lo, V_price_up, y_price, ca_acc, V_adp, V_ca_0)
        constr_top(prob, T, J1, M, V_total, cc_acc, V_adp, V_cc_0, y_top)
        constr_spot_size_high(prob, T, V_spot_up)
        
        if self.options["CAPCOST"]:
            constr_capital_cost(prob, T, v, R, P_cap, s)

        logging.info("Problema definido com sucesso.")
        ###### ######################################

        prob.solve(solver)
        self.model = prob
        self.solver = solver
        logging.info("Problema resolvido com sucesso.")
        return

    def extrair_resultados(self):

        # INDICES E CONJUNTOS
        T, I, J, M, C, J1, N = (
            self.entradas["T"],
            self.entradas["I"],
            self.entradas["J"],
            self.entradas["M"],
            self.entradas["C"],
            self.entradas["J1"],
            self.entradas["N"],
        )

        variables_dict = self.model.variablesDict()

        self.resultado_modelo.update(
            {
                "status": self.model.sol_status,
                "objective_value": float(self.model.objective.value()),
                "solution_time": self.model.solutionTime,
            }
        )

        if self.options["LOCK_POLICY"] == True:
            self.resultado_modelo.update({"y": self.entradas["y"]})

        else:
            self.resultado_modelo.update(
                {
                    "y": otimizador.results_dict_to_list(
                        variables_dict, var="y", index1=J, index2=T
                    )
                }
            )

        self.resultado_modelo.update(
            {"x": otimizador.results_dict_to_list(variables_dict, var="x", index1=T)}
        )

        self.resultado_modelo.update(
            {"s": otimizador.results_dict_to_list(variables_dict, var="s", index1=T)}
        )

        if self.options["BOG_VAR"]:
            self.resultado_modelo.update(
                {
                    "b": otimizador.results_dict_to_list(
                        variables_dict, var="b", index1=T
                    )
                }
            )

        self.resultado_modelo.update(
            {
                "cc_acc": otimizador.results_dict_to_list(
                    variables_dict, var="cc_acc", index1=J1, index2=T
                )
            }
        )
        self.resultado_modelo.update(
            {
                "y_top": otimizador.results_dict_to_list(
                    variables_dict, var="y_top", index1=J1, index2=M, index3=T
                )
            }
        )
        self.resultado_modelo.update(
            {
                "ca_acc": otimizador.results_dict_to_list(
                    variables_dict, var="ca_acc", index1=J1, index2=T
                )
            }
        )
        self.resultado_modelo.update(
            {
                "y_price": otimizador.results_dict_to_list(
                    variables_dict, var="y_price", index1=J1, index2=C, index3=T
                )
            }
        )

        if 1 in self.entradas["Idle"] and self.options["BOG_VAR"] == True:
            self.resultado_modelo.update(
                {
                    "y_bog": otimizador.results_dict_to_list(
                        variables_dict, var="y_bog", index1=T, index2=N
                    )
                }
            )

        if self.options["CAPCOST"]:
            self.resultado_modelo.update(
                {"v": otimizador.results_dict_to_list(variables_dict, var="v", index1=T)}
            )

        # ### TESTE PARA AVALIAR FOBJ DA V2 NA RUN 1 ##
        # self.resultado_modelo.update(
        #         {"v": otimizador.results_dict_to_list(variables_dict, var="v", index1=T)}
        #     )

        # logging.info(f"Extração finalizada. ")

    def calcula_custos(self):

        # INDICES E CONJUNTOS
        T, I, J, M, C, J1, N = (
            self.entradas["T"],
            self.entradas["I"],
            self.entradas["J"],
            self.entradas["M"],
            self.entradas["C"],
            self.entradas["J1"],
            self.entradas["N"],
        )

        # informações de custos da função objetivo
        P_spot = self.entradas["P_spot"]
        custo_fixo = sum(
            [
                pulp.lpSum(self.resultado_modelo["y"][j][t] for j in J)
                * self.entradas["K"]
                for t in T
            ]
        )

        custo_fixo = custo_fixo.value()

        custo_spot = sum([self.resultado_modelo["x"][t] * P_spot[t] for t in T])

        custo_lcp = sum(
            [
                self.entradas["P_spa_price"][(j, c, t)]
                * self.entradas["V_adp"][(j, t)]
                * self.resultado_modelo["y_price"][j][c][t]
                for j in J1
                for c in C
                for t in T
            ]
        )

        custo_demurrage = sum(
            [
                self.resultado_modelo["y"][j][t]
                * self.entradas["P_dem"][j]
                * self.entradas["N_dem"][(j, t)]
                for j in J1
                for t in T
            ]
        )

        custo_top = sum(
            [
                self.resultado_modelo["y_top"][j][m][t]
                * self.entradas["V_adp"][(j, t)]
                * self.entradas["P_spa_top"][(j, m, t)]
                for j in J1
                for m in M
                for t in T
            ]
        )

        if self.options["CAPCOST"]:
            custo_cap = sum(
                [self.entradas["R"] * self.resultado_modelo["s"][t] for t in T]
            )
        else:
            custo_cap = 0

        custo_total = custo_fixo + custo_spot + custo_lcp + custo_demurrage + custo_top

        if self.options["CAPCOST"]:
            custo_total += custo_cap

        custos = {
            "custo_fixo": custo_fixo,
            "custo_spot": custo_spot,
            "custo_lcp": custo_lcp,
            "custo_demurrage": custo_demurrage,
            "custo_top": custo_top,
            "custo_cap": custo_cap,
            "custo_total": custo_total,
        }

        return custos



def visualiza_resultados(opt: otimizador, run, options_df):        
    """Creates a compact operational overview chart."""
    options = options_df.loc[run].to_dict()
    os.makedirs('saidas/graficos_v2', exist_ok=True)

    T = list(opt.entradas["T"])
    J = opt.entradas["J"]
    J1 = range(len(J) - 1)
    I = opt.entradas["I"]
    nome = options["NOME"]

    # Normalized timeline for the horizontal axis
    x = np.arange(len(T))
    demand = np.array([sum(opt.entradas["V_d"][(i, t)] for i in I) for t in T], dtype=float)
    spot = np.array(opt.resultado_modelo["x"], dtype=float)
    inventory = np.array(opt.resultado_modelo["s"], dtype=float)

    palette = {
        "demand": "#cbd5f5",
        "spot": "#d97706",
        "inventory": "#0f766e",
        "bog": "#9333ea",
        "demurrage": "#dc2626"
    }
    spa_colors = ['#2563eb', '#7c3aed', '#0ea5e9', '#4338ca', '#0891b2']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 4.0), dpi=300)  # Scales well when embedded in A4 text

    # Demand envelope
    ax.fill_between(x, 0, demand, color=palette["demand"], alpha=0.45, label='Total Demand', zorder=1)

    bar_width = 1.2
    ax.bar(x, spot, width=bar_width, color=palette["spot"], alpha=0.9, label='Spot Purchases', zorder=2)

    stack_base = spot.copy()
    demurrage_label_shown = False

    for idx, j in enumerate(J1):
        spa_regular = []
        spa_demurrage = []
        planning_marks = []
        prev_plan = 0
        for t in T:
            planned_cargo = opt.entradas['V_adp'][(j, t)]
            planning_marks.append(planned_cargo if planned_cargo and prev_plan == 0 else 0)
            prev_plan = planned_cargo

            purchase = planned_cargo * opt.resultado_modelo["y"][j][t]
            demurrage = opt.entradas['N_dem'][(j, t)] * opt.resultado_modelo["y"][j][t]
            if demurrage > 0:
                spa_demurrage.append(purchase)
                spa_regular.append(0)
            else:
                spa_regular.append(purchase)
                spa_demurrage.append(0)

        spa_regular = np.array(spa_regular, dtype=float)
        spa_demurrage = np.array(spa_demurrage, dtype=float)
        planning_marks = np.array(planning_marks, dtype=float)

        if planning_marks.any():
            ax.vlines(
                x[planning_marks > 0],
                ymin=0,
                ymax=planning_marks[planning_marks > 0],
                colors='#94a3b8',
                linestyles='dashed',
                linewidth=1.2,
                label='SPA Schedule' if idx == 0 else None,
                alpha=0.6,
                zorder=0
            )

        if spa_regular.any():
            ax.bar(
                x,
                spa_regular,
                width=bar_width,
                bottom=stack_base,
                color=spa_colors[idx % len(spa_colors)],
                alpha=0.85,
                label=f'SPA Cargo',
                zorder=3
            )
            stack_base += spa_regular

        demurrage_mask = spa_demurrage > 0
        if demurrage_mask.any():
            ax.bar(
                x[demurrage_mask],
                spa_demurrage[demurrage_mask],
                width=bar_width,
                bottom=stack_base[demurrage_mask],
                color=palette["demurrage"],
                alpha=0.95,
                edgecolor='white',
                linewidth=0.6,
                label='SPA Cargo w/ Demurrage' if not demurrage_label_shown else None,
                zorder=4
            )
            stack_base[demurrage_mask] += spa_demurrage[demurrage_mask]
            demurrage_label_shown = True

    # Inventory line
    ax.plot(x, inventory, color=palette["inventory"], linewidth=1.5, label='Inventory Level', zorder=5)

    # BOG series on secondary axis
    ax2 = None
    if opt.options.get("BOG_VAR", False):
        bog_series = np.array(opt.resultado_modelo["b"], dtype=float)
        if bog_series.any():
            ax2 = ax.twinx()
            ax2.plot(x, bog_series, color=palette["bog"], linewidth=2.2,
                     linestyle='--', label='BOG', zorder=6)
            ax2.set_ylabel('BOG (m³)', fontsize=11, color=palette["bog"])
            ax2.tick_params(axis='y', colors=palette["bog"], labelsize=10)
            ax2.grid(False)

    # General formatting
    tick_step = max(1, len(T)//12)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([str(T[i]) for i in range(0, len(T), tick_step)], rotation=0)
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Volume (m³)', fontsize=12)
    ax.set_title(f'Operational Plan — {nome} (Run {run})', fontsize=12, pad=20, fontweight='bold')

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2

    ax.legend(
        handles,
        labels,
        loc='upper left',
        bbox_to_anchor=(0, 1.12),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.0,
        handlelength=1.6
    )

    plt.savefig(f'saidas/graficos_v2/{nome}_run{run}_estoque_compras.png', bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()