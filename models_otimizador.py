import pulp
import numpy as np
import logging
import pandas as pd
import json





class otimizador:
    """Função para definição do otimizador ee xtração de resultados"""

    def __init__(self, nome, arquivo_json, arquivo_excel, options) -> None:

        self.nome = nome
        self.timeLimit = options["TIMELIMIT"]
        self.options = options
        logging.info(f"Carregando parâmetros para o cenário {nome}.")
        self.entradas = self.carregar_parametros(nome, arquivo_json, arquivo_excel)
        logging.info(f"Parâmetros carregados para o cenário {nome}. Iniciando otimização.")
        # self.model = self.solve_model()
        
        self.resultado_modelo = {
            "status": "",
            "objective_value": 0,
            "solution_time": 0,
            "d": [],
            "rv": [],
            "x": [],
            "y": [],
            "k": [],
            "s": [],
            "s_min": [],
            "b": [],
            "z": [],
            "cc_acc": [],
            "topc": [],
            "ca_acc": [],
            "lgc": [],
            "f": [],
            "fy": [],
            "by": [],
        }
        
    @staticmethod
    def results_dict_to_list(variables_dict, var: str, index1, index2=None, index3=None):
        logging.info(f"Extraindo resultados para a variável {var}.")
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
                [variables_dict[f"{var}_{j}_{m}"].value() for m in index2] for j in index1
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
        

        with open(arquivo_json, 'r', encoding='utf-8') as f:
            dados = json.load(f)[nome]


        params_por_contrato = dados['parametros_por_contrato']
        indices = dados['indices']
        conjuntos = dados['conjuntos']
        parametros_modelo = dados['parametros_modelo']
        variaveis_fixadas = dados['variaveis_fixadas']

        entradas = {}
        entradas.update({key: range(value) for key, value in indices.items()})
        entradas.update({key: value for key, value in variaveis_fixadas.items()})
        entradas.update(conjuntos)
        entradas.update(parametros_modelo)
        entradas["K"] = entradas["K"] * entradas["DOLAR"]
        entradas["PDEM"] = entradas["PDEM"] * entradas["DOLAR"]
        if self.options["BOG_TEST"]:
            entradas["x"] = variaveis_fixadas["x"]
            entradas["y"] = variaveis_fixadas["y"]

        for j, contrato in params_por_contrato.items():
            for param, valor in contrato.items():
                if param in ['TPH', 'TPL', 'LCH', 'LCL']:
                    entradas[param] = {}
                    for k, v in enumerate(valor):
                        entradas[param][(int(j), k)] = v
                else:
                    entradas[param] = {int(j): valor}

        # Aplicar conversão nos dados do Excel

        params_tuplas = otimizador.converter_excel_para_tuplas(arquivo_excel,aba_nome=nome)
        entradas.update(params_tuplas)
        
        INT = {}
        for j in range(len(entradas["J"]) - 1):
            int_ = []
            ini = 0
            fim = 0
            i_1 = 0
            T = entradas['T']
            for t in T:
                i_0 = entradas['W'][(j,t)]
                if i_0 ==0 and i_1 ==0:
                    pass
                elif i_0 !=0 and i_1 ==0:
                    ini = t
                elif i_0==0 and i_1 !=0:
                    fim = t-1
                    int_.append([ini,fim])
                else:
                    pass
                i_1 = i_0
                    
            INT[j] = np.array(int_)   
        entradas.update({"INT": INT})
                    
        return entradas
    
    
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
                raise ValueError(f"Aba '{aba_nome}' não encontrada. Abas disponíveis: {available_sheets}")
            else:
                raise ValueError(f"Erro ao ler arquivo: {e}")
            
        params_convertidos = {}
        
        # Separar por tipo de parâmetro
        for coluna in df.columns:
            if coluna.startswith('D_'):
                # Parâmetro D indexado por (i,t)
                i = int(coluna.split('_')[1])  # Extrair índice i do nome da coluna
                if 'D' not in params_convertidos:
                    params_convertidos['D'] = {}
                
                for t, valor in parametros[coluna].items():
                    params_convertidos['D'][(i, t)] = valor
                    
            elif coluna.startswith('W_'):
                # Parâmetro W indexado por (j,t)
                j = int(coluna.split('_')[1])  # Extrair índice j do nome da coluna
                if 'W' not in params_convertidos:
                    params_convertidos['W'] = {}
                
                for t, valor in parametros[coluna].items():
                    params_convertidos['W'][(j, t)] = valor
                    
            elif coluna.startswith('DEM_'):
                # Parâmetro DEM indexado por (j,t)
                j = int(coluna.split('_')[1])
                if 'DEM' not in params_convertidos:
                    params_convertidos['DEM'] = {}
                
                for t, valor in parametros[coluna].items():
                    params_convertidos['DEM'][(j, t)] = valor
                    
            elif coluna.startswith('Y_EXPORT_'):
                # Parâmetro Y_EXPORT indexado por (j,t)
                j = int(coluna.split('_')[2])
                if 'Y_EXPORT' not in params_convertidos:
                    params_convertidos['Y_EXPORT'] = {}
                
                for t, valor in parametros[coluna].items():
                    params_convertidos['Y_EXPORT'][(j, t)] = valor
                    
            elif coluna.startswith('TPT_'):
                # Parâmetro TPT indexado por (j,t)
                j = int(coluna.split('_')[1])
                m = int(coluna.split('_')[2])
                if 'TPT' not in params_convertidos:
                    params_convertidos['TPT'] = {}

                for t, valor in parametros[coluna].items():
                    params_convertidos['TPT'][(j,m, t)] = valor
                    
            elif coluna.startswith('LCP_'):
                # Parâmetro LCP indexado por (j,m,t)
                j = int(coluna.split('_')[1])
                c = int(coluna.split('_')[2])
                if 'LCP' not in params_convertidos:
                    params_convertidos['LCP'] = {}

                for t, valor in parametros[coluna].items():
                    params_convertidos['LCP'][(j,c, t)] = valor
                    
            elif coluna == 'IDLE':
                # Parâmetro IDLE indexado por t
                if 'IDLE' not in params_convertidos:
                    params_convertidos['IDLE'] = {}
                
                for t, valor in parametros[coluna].items():
                    params_convertidos['IDLE'][t] = valor
                    
            elif coluna == 'PS':
                # Parâmetro PS indexado por t
                if 'PS' not in params_convertidos:
                    params_convertidos['PS'] = {}
                
                for t, valor in parametros[coluna].items():
                    params_convertidos['PS'][t] = valor
        
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
        T, I, J, M, C, TP, CF, N = (
            self.entradas["T"],
            self.entradas["I"],
            self.entradas["J"],
            self.entradas["M"],
            self.entradas["C"],
            self.entradas["TP"],
            self.entradas["CF"],
            self.entradas["N"],
        )
        J1 = range(len(self.entradas["J"]) - 1)
        
        # PARAMETROS DO MODELO
        K, H, S0, SW, S_MAX, S_FLEX, S_MIN_IDLE, S_MIN_PROD,Q1, Q2, PDEM, SMIN_FLEX, BOGCTE, DOLAR, VH, VL, BI, BP, BG = (
            self.entradas["K"],
            self.entradas["H"],
            self.entradas["S0"],
            self.entradas["SW"],
            self.entradas["S_MAX"],
            self.entradas["S_FLEX"],
            self.entradas["S_MIN_IDLE"],
            self.entradas["S_MIN_PROD"],
            self.entradas["Q1"],
            self.entradas["Q2"],
            self.entradas["PDEM"],
            self.entradas["SMIN_FLEX"],
            self.entradas["BOGCTE"],
            self.entradas["DOLAR"],
            self.entradas["VH"],
            self.entradas["VL"],
            self.entradas["BI"],
            self.entradas["BP"],
            self.entradas["BG"]
        )
        PS_MEAN = np.array(list(self.entradas["PS"].values())).mean()
        F_MIN = 0
        F_MAX = 0
        
        # PARAMETROS POR CONTRATO
        INT, VT, CC_INI, CA_INI, TPH, TPL, LCH, LCL = (
            self.entradas["INT"],
            self.entradas["VT"],
            self.entradas["CC_INI"],
            self.entradas["CA_INI"],
            self.entradas["TPH"],
            self.entradas["TPL"],
            self.entradas["LCH"],
            self.entradas["LCL"],
        )
        for j in INT:
            INT[j] = np.array(INT[j])

        # PARAMETROS INDEXADOS NO TEMPO
        D, W, IDLE, DEM, PS, Y_EXPORT, TPT, LCP = (
            self.entradas["D"],
            self.entradas["W"],
            self.entradas["IDLE"],
            self.entradas["DEM"],
            self.entradas["PS"],
            self.entradas["Y_EXPORT"],
            self.entradas["TPT"],
            self.entradas["LCP"],
        )



        logging.info("Iniciando definição do modelo de otimização.")
       
        # # BP = self.params_operacao["BOG_PRODUCTION_m³GNL"]
        # PS_MEAN = np.array(list(self.params_modelo["NR_DICT_PRECO_CONTRATO_SPOT"].values())).mean()

        ## VARIAVEIS
        
        if self.options["BOG_TEST"] == False:
            y = pulp.LpVariable.dicts("y", (J, T), cat="Binary")
            
        else:
            # x = self.entradas["x"]
            y = self.entradas["y"]
            
        x = pulp.LpVariable.dicts("x", T, lowBound=0, upBound=Q2)
        logging.info("Variáveis de decisão x e y definidas.")
        s = pulp.LpVariable.dicts("s", T, lowBound=0, upBound=S_MAX)
        # s_min = pulp.LpVariable.dicts("s_min", T, lowBound=0, upBound=S_FLEX)
        b = pulp.LpVariable.dicts("b", T, lowBound=0)
        by = pulp.LpVariable.dicts("by", (T, N), cat="Binary")
        cc_acc = pulp.LpVariable.dicts("cc_acc", (TP, T), lowBound=0)
        topc = pulp.LpVariable.dicts("topc", (TP, M, T), cat="Binary")
        ca_acc = pulp.LpVariable.dicts("ca_acc", (J1, T), lowBound=0)
        lgc = pulp.LpVariable.dicts("lgc", (J1, C, T), cat="Binary")
        
        
        f = pulp.LpVariable.dicts("f", (CF, T), lowBound=0)
        fy = pulp.LpVariable.dicts("fy", (CF, C, T), lowBound=0)
        # d = pulp.LpVariable.dicts("d", (J, I, T), lowBound=0)
        # rv = pulp.LpVariable.dicts("rv", (J, T), lowBound=0)
        # k = pulp.LpVariable.dicts("k", (J, T), cat="Binary")
        # z = pulp.LpVariable.dicts("z", T, cat="Binary")
        # if self.options["BOG_TEST"]:
            
            
            
        # DEFINICAO DO PROBLEMA
        
        solver = pulp.PULP_CBC_CMD(timeLimit=self.timeLimit,logPath="cbc.log")
        # solver = pulp.PULP_CBC_CMD(threads = 6, timeLimit= 60)
        prob = pulp.LpProblem("Simulacao_cenarios", pulp.LpMinimize)
        
        # FUNCAO OBJETIVO
        def objetive_func():
            total = 0
            for t in T:
                total += pulp.lpSum(y[j][t] for j in J) * K # custo fixo
                # total += H * s[t] * PS_MEAN # custo de holding
                total += x[t] * PS[(t)] # custo compra spot
                total += pulp.lpSum(
                    LCP[(j, c, t)] * W[(j, t)] * lgc[j][c][t] for j in TP for c in C
                ) # custo compra LCP 
                total += pulp.lpSum(y[j][t] * PDEM * DEM[(j, t)] for j in TP) # custo demurrage
                total += pulp.lpSum(
                    topc[j][m][t] * W[(j, t)] * TPT[(j, m, t)] for j in TP for m in M
                ) # custo take or pay
                # total += sum(DW[(j, i)] * d[j][i][t] * PS_MEAN for i in I for j in J)
                # total += s_min[t] * SW * PS_MEAN
                # total += z[t] * PSPEQ
                # total += pulp.lpSum(k[j][t] * PLOCK for j in TP)
                # total += pulp.lpSum(fy[j][c][t] * LCP[(j, c, t)] for j in CF for c in C)
            return total

        prob += objetive_func(), "Custo Total"

        # @staticmethod
        # def add_restricoes_balanco_inventario_bog_cte(prob, s, x, y, f, D, W, T, I, CF, TP, S0):

        # Balanço de massa sem considerar bog
        def constr_store_balance_no_bog(prob, T, s, x, y, f, D, W, I, CF, TP, S0):
            for t in T:
                if t == 0:
                    prob += (s[t] == S0 + x[t]  + pulp.lpSum(
                        W[(j, t)] * y[j][t] for j in TP
                    ) - pulp.lpSum(D[(i, t)] for i in I),  f"Store_Balance_Bog_Cte_Constraint_{t}")
                    # + pulp.lpSum(f[j][t] for j in CF)
                else:
                    prob += (s[t] == s[t - 1] + x[t]  + pulp.lpSum(W[(j, t)] * y[j][t] for j in TP) - pulp.lpSum(
                        D[(i, t)] for i in I
                    ) ,  f"Store_Balance_Bog_Cte_Constraint_{t}")
                    # + pulp.lpSum(
                    #     f[j][t] for j in CF
                    # )
                    


        def constr_store_balance_bog(prob, T, s, x, y, f, D, W, I, CF, TP, S0, b, IDLE, BP):
            for t in T:
                if t == 0:
                    prob += (s[t] == S0 + x[t]  + pulp.lpSum(
                        W[(j, t)] * y[j][t] for j in TP
                    ) - pulp.lpSum(D[(i, t)] for i in I) - b[t] * IDLE[t] - BP * (1 - IDLE[t]),  f"Store_Balance_Bog_Cte_Constraint_{t}")
                # + pulp.lpSum(f[j][t] for j in CF)
                else:
                    prob += (s[t] == s[t - 1] + x[t]  + pulp.lpSum(W[(j, t)] * y[j][t] for j in TP) - pulp.lpSum(
                        D[(i, t)] for i in I
                    ) - b[t] * IDLE[t] - BP * (1 - IDLE[t]),  f"Store_Balance_Bog_Cte_Constraint_{t}")
                # + pulp.lpSum(
                #         f[j][t] for j in CF
                #     )

                # Tipo de modelo
        def constr_bog_level(prob, T, IDLE, s, by, VL, VH, S_MAX, b, BI, BP):
            for t in T:
                if IDLE[t] == 1:
                    for n in N:
                        prob += VL[n] * by[t][n] - s[t] <= 0, f"Boglevel_Lower_{t}_{n}"
                        prob += (
                        VH[n] * by[t][n] + S_MAX * (1 - by[t][n]) - s[t] >= 0,
                        f"Boglevel_Upper_{t}_{n}",
                    )
                        
                    prob += sum(by[t][n] for n in N) == 1, f"Boglevel_Sum_{t}"
                    prob += b[t] == sum(BI[n] * by[t][n] for n in N), f"Boglevel_b_{t}"
                
                else:
                    prob += b[t] == BP, f"Boglevel_b_fixed_{t}"
                    prob += sum(by[t][n] for n in N) == 0, f"Boglevel_by_fixed_{t}"

        def constr_bog_constante(prob, T, IDLE, b, BG, BP):
            for t in T:
                if IDLE[t] == 1:
                    prob += b[t] == BG, f"BogConstante_b_{t}"
                else:
                    prob += b[t] == BP, f"BogConstante_b_fixed_{t}"

        
        # Inventário IDLE
        def constr_store_capacity_lower_idle(prob, T, S_MIN_IDLE):
            for t in T:
                prob += (
                    s[t] >= S_MIN_IDLE * IDLE[t],
                    f"Store_Capacity_L_Idle_Constraint_{t}",
                )
                
        

        def constr_store_capacity_lower_prod(prob, T, S_MIN_PROD, SMIN_FLEX, s, IDLE):
            for t in T:
                if SMIN_FLEX:
                    prob += (
                        s[t] >= S_MIN_PROD * (1 - IDLE[t]) ,
                        f"Store_Capacity_L_Prod_Constraint_{t}",
                    )
                    # - s_min[t]
                else:
                    prob += (
                        s[t] >= S_MIN_PROD * (1 - IDLE[t]),
                        f"Store_Capacity_L_Prod_Constraint_{t}",
                    )


        # Intervalos de carga
        def constr_gap_between_arrival(prob, T, J, y):
            for t in range(len(T)):
                if t == 0:
                    prob += (
                        sum(y[j][t] for j in range(len(J))) <= 1,
                        f"Interval_Constraint_{t}",
                    )
                elif t == 1:
                    prob += (
                        sum(y[j][t] + y[j][t - 1] for j in range(len(J))) <= 1,
                        f"Interval_Constraint_{t}",
                    )
                elif t == 2:
                    prob += (
                        sum(y[j][t] + y[j][t - 1] + y[j][t - 2] for j in range(len(J)))
                        <= 1,
                        f"Interval_Constraint_{t}",
                    )
                else:
                    prob += (
                        sum(
                            y[j][t] + y[j][t - 1] + y[j][t - 2] + y[j][t - 3]
                            for j in range(len(J))
                        )
                        <= 1,
                        f"Interval_Constraint_{t}",
                    )

        # Delimitar a compra de uma única carga por intervalo definido
        def constr_interval_limit_one_load(prob, T, TP, INT, y):
            constr_demurrage_interval = []
            for j in TP:
                for interval in INT[j]:
                    constr_demurrage_interval.append(
                        prob.addConstraint(
                            pulp.lpSum(y[j][t] for t in range(interval[0], interval[1])) <= 1,
                            f"Demurrage_Interval_Constraint_{j}_{interval[0]}_{interval[1]}",
                        )
                    )
                    
        # Lng Cost
        def constr_lng_cost(prob, T, J1, TP, C, LCL, LCH, lgc, ca_acc, W, CA_INI, f):
            constr_lgc_faixa = []
            for t in T:
                for j in TP:
                    for c in C:
                        constr_lgc_faixa.append(LCL[(j, c)] * lgc[j][c][t] - ca_acc[j][t] <= 0)
                        constr_lgc_faixa.append(
                            LCH[(j, c)] * lgc[j][c][t]
                            + VT[j] * (1 - lgc[j][c][t])
                            - ca_acc[j][t]
                            >= 0
                        )
                    if t == 0:
                        constr_lgc_faixa.append(
                            ca_acc[j][t]
                            == pulp.lpSum([lgc[j][c][t] for c in C]) * W[(j, t)] + CA_INI[j]
                        )
                    else:
                        constr_lgc_faixa.append(
                            ca_acc[j][t]
                            == ca_acc[j][t - 1]
                            + pulp.lpSum(lgc[j][c][t] for c in C) * W[(j, t)]
                        )
                for j in CF:
                    for c in C:
                        constr_lgc_faixa.append(LCL[(j, c)] * lgc[j][c][t] - ca_acc[j][t] <= 0)
                        constr_lgc_faixa.append(
                            LCH[(j, c)] * lgc[j][c][t]
                            + VT[j] * (1 - lgc[j][c][t])
                            - ca_acc[j][t]
                            >= 0
                        )
                    if t == 0:
                        constr_lgc_faixa.append(ca_acc[j][t] == f[j][t] + CA_INI[j])
                    else:
                        constr_lgc_faixa.append(ca_acc[j][t] == ca_acc[j][t - 1] + f[j][t])
            for idx, constraint in enumerate(constr_lgc_faixa):
                prob += constraint, f"Lgc_Faixa_Constraint_{idx}"
            for j in J1:
                for t in T:
                    prob += (
                        pulp.lpSum(lgc[j][c][t] for c in C) == y[j][t],
                        f"lgc_y_constr_{j}_{t}",
                    )
        
        def constr_top(prob, T, TP, M, VT, cc_acc, W, CC_INI, topc):           
            # Take or pay
            constr_topc_faixa = []
            for t in T:
                for j in TP:
                    for m in M:
                        constr_topc_faixa.append(
                            TPL[(j, m)] * topc[j][m][t] - cc_acc[j][t] <= 0
                        )
                        constr_topc_faixa.append(
                            TPH[(j, m)] * topc[j][m][t]
                            + VT[j] * (1 - topc[j][m][t])
                            - cc_acc[j][t]
                            >= 0
                        )
                    if t == 0:
                        constr_topc_faixa.append(
                            cc_acc[j][t]
                            == pulp.lpSum(topc[j][m][t] for m in M) * W[(j, t)] + CC_INI[j]
                        )
                    else:
                        constr_topc_faixa.append(
                            cc_acc[j][t]
                            == cc_acc[j][t - 1]
                            + pulp.lpSum([topc[j][m][t] for m in M]) * W[(j, t)]
                        )
            for idx, constraint in enumerate(constr_topc_faixa):
                prob += constraint, f"Topc_Faixa_Constraint_{idx}"

            # def constr_topc_within_interval(prob):
            for j in TP:
                for interval in INT[j]:
                    for t in range(interval[0], interval[1]):
                        if t == interval[0]:
                            prob += (
                                pulp.lpSum(topc[j][m][t] for m in M)
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
                                    topc[j][m][t]
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

        def contr_flex_fee(prob, T, CF, C, F_MIN, F_MAX, f, fy, lgc):
            # Flex fee
            constr_flex_fee = []
            for j in CF:
                for t in T:
                    for c in C:
                        constr_flex_fee.append(
                            f[j][t] - F_MAX[j] * (1 - lgc[j][c][t]) <= fy[j][c][t]
                        )
                        constr_flex_fee.append(
                            f[j][t] + F_MIN[j] * (1 - lgc[j][c][t]) >= fy[j][c][t]
                        )
                        constr_flex_fee.append(F_MIN[j] * (lgc[j][c][t]) <= fy[j][c][t])
                        constr_flex_fee.append(F_MAX[j] * (lgc[j][c][t]) >= fy[j][c][t])
            for j in CF:
                for t in T:
                    constr_flex_fee.append(
                        F_MIN[j] * pulp.lpSum(lgc[j][c][t] for c in C) <= f[j][t]
                    )
                    constr_flex_fee.append(
                        F_MAX[j] * pulp.lpSum(lgc[j][c][t] for c in C) >= f[j][t]
                    )
            for idx, constraint in enumerate(constr_flex_fee):
                prob += constraint, f"flex_fee_Constraint_{idx}"
        
            
        # Spot
        def constr_spot_size_high(prob, T, Q2):
            for t in T:
                prob += x[t] <= Q2 * y[len(J) - 1][t], f"Spot_Constraint_{t}"

        
       
        logging.info("Definição das restrições do modelo de otimização.")   
        # def constr_spot_size_lower(prob, T, Q1, Q2):
        #     for t in T:
        #         prob += (
        #             x[t] - Q1 * y[len(J) - 1][t] <= Q2 * (1 - z[t]),
        #             f"Spot_Size_Lower_Constraint_1_{t}",
        #         )
        #         prob += (
        #             Q1 * y[len(J) - 1][t] - x[t] <= Q2 * z[t],
        #             f"Spot_Size_Lower_Constraint_2_{t}",
        #         )

        
        # Montar problema
        if self.options["NO_BOG"]:
            constr_store_balance_no_bog(prob, T, s, x, y, f, D, W, I, CF, TP, S0)
            
        elif self.options["BOG_CTE"]:
            constr_store_balance_bog(prob, T, s, x, y, f, D, W, I, CF, TP, S0, b, IDLE, BP)
            constr_bog_constante(prob, T, IDLE, b, BG, BP)
        
        elif self.options["BOG_VAR"]:
            constr_store_balance_bog(prob, T, s, x, y, f, D, W, I, CF, TP, S0, b, IDLE, BP)
            constr_bog_level(prob, T, IDLE, s, by, VL, VH, S_MAX, b, BI, BP)
        
        
        constr_store_capacity_lower_idle(prob, T, S_MIN_IDLE)
        constr_store_capacity_lower_prod(prob, T, S_MIN_PROD, SMIN_FLEX, s, IDLE,)
        constr_gap_between_arrival(prob, T, J, y)
        constr_interval_limit_one_load(prob, T, TP, INT, y)
        constr_lng_cost(prob, T, J1, TP, C, LCL, LCH, lgc, ca_acc, W, CA_INI, f)
        constr_top(prob, T, TP, M, VT, cc_acc, W, CC_INI, topc)
        # contr_flex_fee(prob, T, CF, C, F_MIN, F_MAX, f, fy, lgc)
        constr_spot_size_high(prob, T, Q2)
        # constr_spot_size_lower(prob, T, Q1, Q2)

        logging.info("Iniciando resolução do modelo de otimização.")
        prob.solve(solver)
        logging.info("Resolução finalizada.")
        self.model = prob
        self.solver = solver
        return 


        


    def extrair_resultados(self):
        
        # INDICES E CONJUNTOS
        T, I, J, M, C, TP, CF, N = (
            self.entradas["T"],
            self.entradas["I"],
            self.entradas["J"],
            self.entradas["M"],
            self.entradas["C"],
            self.entradas["TP"],
            self.entradas["CF"],
            self.entradas["N"],
        )
        J1 = range(len(self.entradas["J"]) - 1)
        

        variables_dict = self.model.variablesDict()

        self.resultado_modelo.update({"status": self.model.sol_status,
                                      "objective_value": float(self.model.objective.value()),
                                      "solution_time": self.model.solutionTime})
        
        if self.options["BOG_TEST"] == True:
            self.resultado_modelo.update({
                # 'x': self.entradas["x"],
                                          'y': self.entradas["y"]})
                
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
        # self.resultado_modelo.update(
        #     {
        #         "s_min": otimizador.results_dict_to_list(
        #             variables_dict, var="s_min", index1=T
        #         )
        #     }
        # )
        if self.options["NO_BOG"] == False:
            self.resultado_modelo.update(
                {"b": otimizador.results_dict_to_list(variables_dict, var="b", index1=T)}
            )

        self.resultado_modelo.update(
            {
                "cc_acc": otimizador.results_dict_to_list(
                    variables_dict, var="cc_acc", index1=TP, index2=T
                )
            }
        )
        self.resultado_modelo.update(
            {
                "topc": otimizador.results_dict_to_list(
                    variables_dict, var="topc", index1=TP, index2=M, index3=T
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
                "lgc": otimizador.results_dict_to_list(
                    variables_dict, var="lgc", index1=J1, index2=C, index3=T
                )
            }
        )
        # self.resultado_modelo.update(
        #     {
        #         "f": otimizador.results_dict_to_list(
        #             variables_dict, var="f", index1=CF, index2=T
        #         )
        #     }
        # )
        # self.resultado_modelo.update(
        #     {
        #         "fy": otimizador.results_dict_to_list(
        #             variables_dict, var="fy", index1=CF, index2=C, index3=T
        #         )
        #     }
        # )

        if (
            1 in self.entradas["IDLE"]
            and self.options["BOG_VAR"] == True
        ):
            self.resultado_modelo.update(
                {
                    "by": otimizador.results_dict_to_list(
                        variables_dict, var="by", index1=T, index2=N
                    )
                }
            )

        
        logging.info(f"Extração finalizada. ")
        
        
 