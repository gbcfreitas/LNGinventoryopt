import pulp
import numpy as np
import logging
import pandas as pd
import json
       


class otimizador:
    """Função para definição do otimizador ee xtração de resultados"""

    def __init__(self, params_modelo, params_cenario, number_index, dolar,timeLimit,params_operacao,) -> None:
        self.params_cenario = params_cenario
        self.number_index = number_index
        self.params_modelo = params_modelo
        self.dolar = dolar
        self.timeLimit = timeLimit
        self.params_operacao = params_operacao
        self.model = self.solve_model()
        
        self.resultado_modelo = {
            "status": "",
            "objective_value": 0,
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
    def converter_excel_para_tuplas(arquivo_excel, aba_nome=None):
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

    def carregar_parametros(self, nome, arquivo_json, arquivo_excel):
        # arquivo_excel = 'df_params.xlsx'
        # arquivo_json = 'cen1.json'
        # aba_nome = 'Cen1'
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

        entradas = {}
        entradas.update({key: range(value) for key, value in indices.items()})
        entradas.update(conjuntos)
        entradas.update(parametros_modelo)
        entradas["K"] = entradas["K"] * entradas["DOLAR"]
        entradas["PDEM"] = entradas["PDEM"] * entradas["DOLAR"]


        for j, contrato in params_por_contrato.items():
            for param, valor in contrato.items():
                if param in ['TPH', 'TPL', 'LCH', 'LCL']:
                    entradas[param] = {}
                    for k, v in enumerate(valor):
                        entradas[param][(j, k)] = v
                else:
                    entradas[param] = {j: valor}

        # Aplicar conversão nos dados do Excel

        params_tuplas = otimizador.converter_excel_para_tuplas(arquivo_excel, aba_nome=aba_nome)
        entradas.update(params_tuplas)

        return entradas

    @staticmethod
    def results_dict_to_list(variables_dict, var: str, index1, index2=None, index3=None):
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

    solver = pulp.PULP_CBC_CMD(timeLimit=self.timeLimit)
    # solver = pulp.PULP_CBC_CMD(threads = 6, timeLimit= 60)
    prob = pulp.LpProblem("Simulacao_cenarios", pulp.LpMinimize)

    ##Indices e conjuntos
    T = self.number_index
    I = range(self.params_modelo["NR_QUANTIDADE_DE_DEMANDAS"])
    J = range(self.params_modelo["NR_QUANTIDADE_DE_CONTRATOS"])
    N = range(7)
    M = range(self.params_modelo["NR_MAX_CATEGORIAS_TOP"])
    C = range(self.params_modelo["NR_MAX_CATEGORIAS_LNGCOST"])
    TP = self.params_modelo["NR_LIST_CONJ_CONTRATOS_TOP"]
    CF = self.params_modelo["NR_LIST_CONJ_CONTRATOS_FF"]
    J1 = range(self.params_modelo["NR_QUANTIDADE_DE_CONTRATOS"] - 1)

    ### Parametros indexados
    D = self.params_modelo["NR_DICT_VALOR_DEMANDA"]
    W = self.params_modelo["NR_DICT_CARGAS_PROGRAMADAS_TOP"]
    IDLE = self.params_modelo["NR_LIST_STATUS_IDLE"]
    DEM = self.params_modelo["NR_DICT_DIAS_DEMURRAGE"]
    INT = self.params_modelo["NR_DICT_INTERVALOS_DE_CARGAS"]
    TPH = self.params_modelo["NR_DICT_QTDD_SUPERIOR_CATEG_TOP"]
    TPL = self.params_modelo["NR_DICT_QTDD_INFERIOR_CATEG_TOP"]
    TPT = self.params_modelo["NR_DICT_PRECO_CATEG_TOP"]
    VT = self.params_modelo["NR_DICT_VOLUME_TOTAL_CONTRATO"]
    CC_INI = self.params_modelo["NR_DICT_QTDD_CANCELADA_INI"]
    LCH = self.params_modelo["NR_DICT_QTDD_SUPERIOR_CATEG_LNGCOST"]
    LCL = self.params_modelo["NR_DICT_QTDD_INFERIOR_CATEG_LNGCOST"]
    LCP = self.params_modelo["NR_DICT_PRECO_CATEG_LNGCOST"]
    PS = self.params_modelo["NR_DICT_PRECO_CONTRATO_SPOT"]
    CA_INI = self.params_modelo["NR_DICT_QTDD_CONFIRMADA_INI"]
    F_MIN = self.params_modelo["NR_DICT_VOLUME_MINIMO_FF"]
    F_MAX = self.params_modelo["NR_DICT_VOLUME_MAXIMO_FF"]
    N_CEN = self.params_modelo["NR_QUANTIDADE_DE_DIAS"]
    
    # Y_LOCKED = self.params_modelo["BOOL_CARGA_TRANCADA"]
    # DW = self.params_modelo["NR_DICT_PESO_DEMANDA_CONTRATO"]
    # Y_EXPORT = self.params_modelo["NR_DICT_CARGA_EXPORT"]

    ### Parametros únicos

    K = self.params_cenario["K"] * self.dolar
    H = self.params_cenario["H"]
    S0 = self.params_cenario["S0"]
    SW = self.params_cenario["SW"]
    S_MAX = self.params_cenario["S_MAX"]
    S_FLEX = self.params_cenario["S_FLEX"]
    S_MIN_IDLE = self.params_cenario["S_MIN_IDLE"]
    S_MIN_PROD = self.params_cenario["S_MIN_PROD"]
    # BP = self.params_operacao["BOG_PRODUCTION_m³GNL"]
    # BG = self.params_cenario["BG"]
    # Q1 = self.params_cenario["Q1"]
    Q2 = self.params_cenario["Q2"]
    PDEM = self.params_cenario["PDEM"] * self.dolar
    # PSPEQ = self.params_cenario["PDEM"] * 200 * self.dolar
    # PLOCK = self.params_cenario["PDEM"] * 100000 * self.dolar
    SMIN_FLEX = self.params_cenario["SMIN_FLEX"]
    BOGCTE = self.params_cenario["BOGCTE"]
    PS_MEAN = np.array(list(self.params_modelo["NR_DICT_PRECO_CONTRATO_SPOT"].values())).mean()

    ## Variáveis
    d = pulp.LpVariable.dicts("d", (J, I, T), lowBound=0)
    rv = pulp.LpVariable.dicts("rv", (J, T), lowBound=0)
    x = pulp.LpVariable.dicts("x", T, lowBound=0, upBound=Q2)
    y = pulp.LpVariable.dicts("y", (J, T), cat="Binary")
    k = pulp.LpVariable.dicts("k", (J, T), cat="Binary")
    s = pulp.LpVariable.dicts("s", T, lowBound=0, upBound=S_MAX)
    s_min = pulp.LpVariable.dicts("s_min", T, lowBound=0, upBound=S_FLEX)
    b = pulp.LpVariable.dicts("b", T, lowBound=0)
    by = pulp.LpVariable.dicts("by", (T, N), cat="Binary")
    z = pulp.LpVariable.dicts("z", T, cat="Binary")
    cc_acc = pulp.LpVariable.dicts("cc_acc", (TP, T), lowBound=0)
    topc = pulp.LpVariable.dicts("topc", (TP, M, T), cat="Binary")
    ca_acc = pulp.LpVariable.dicts("ca_acc", (J1, T), lowBound=0)
    lgc = pulp.LpVariable.dicts("lgc", (J1, C, T), cat="Binary")
    f = pulp.LpVariable.dicts("f", (CF, T), lowBound=0)
    fy = pulp.LpVariable.dicts("fy", (CF, C, T), lowBound=0)

    ## valores para restrição
    VH = np.array([24273, 48646.6, 72919.6, 97293.2, 121566.2, 145939.8, 170212.8])
    VL = np.array([0, 24273, 48646.6, 72919.6, 97293.2, 121566.2, 145939.8])
    BI = np.array(
        [29.1276, 58.37592, 87.50352, 116.75184, 145.87944, 175.12776, 204.25536]
    )

    ## Função objetivo
    def objetive_func():
        total = 0
        for t in T:
            total += pulp.lpSum(y[j][t] for j in J) * K
            total += H * s[t] * PS_MEAN
            total += x[t] * PS[(t)]
            total += pulp.lpSum(
                LCP[(j, c, t)] * W[(j, t)] * lgc[j][c][t] for j in TP for c in C
            )
            total += pulp.lpSum(y[j][t] * PDEM * DEM[(j, t)] for j in TP)
            total += pulp.lpSum(
                topc[j][m][t] * W[(j, t)] * TPT[(j, m, t)] for j in TP for m in M
            )
            # total += sum(DW[(j, i)] * d[j][i][t] * PS_MEAN for i in I for j in J)
            total += s_min[t] * SW * PS_MEAN
            # total += z[t] * PSPEQ
            # total += pulp.lpSum(k[j][t] * PLOCK for j in TP)
            total += pulp.lpSum(fy[j][c][t] * LCP[(j, c, t)] for j in CF for c in C)
        return total

    prob += objetive_func(), "Custo Total"

    # Restrições

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

    # Lng Cost
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

    def constr_topc_within_interval(prob):
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

    constr_topc_within_interval(prob)

    # Balanço de massa
    def constr_store_balance(t):
        if t == 0:
            return s[t] == S0 + x[t] + pulp.lpSum(f[j][t] for j in CF) + pulp.lpSum(
                W[(j, t)] * y[j][t] for j in TP
            ) - pulp.lpSum(D[(i, t)] for i in I) 
            
            # - b[t] * IDLE[t] - BP * (1 - IDLE[t])
        else:
            return s[t] == s[t - 1] + x[t] + pulp.lpSum(
                f[j][t] for j in CF
            ) + pulp.lpSum(W[(j, t)] * y[j][t] for j in TP) - pulp.lpSum(
                D[(i, t)] for i in I
            ) 
            # - b[t] * IDLE[t] - BP * (1 - IDLE[t])

    for t in T:
        prob += constr_store_balance(t), f"Store_Balance_Constraint_{t}"

    # Reservatório Virtual
    # def constr_reserv_virtual(prob):
    #     for t in T:
    #         for j in J:
    #             if j == len(J) - 1:
    #                 if t == 0:
    #                     prob += (
    #                         rv[j][t] == S0 + x[t] - pulp.lpSum(d[j][i][t] for i in I),
    #                         f"Reserv_Virtual_Constraint_Last_{j}_{t}",
    #                     )
    #                 else:
    #                     prob += (
    #                         rv[j][t]
    #                         == rv[j][t - 1] + x[t] - pulp.lpSum(d[j][i][t] for i in I),
    #                         f"Reserv_Virtual_Constraint_Last_{j}_{t}",
    #                     )
    #             elif j in TP:
    #                 if t == 0:
    #                     prob += (
    #                         rv[j][t]
    #                         == S0
    #                         + W[(j, t)] * y[j][t]
    #                         - pulp.lpSum(d[j][i][t] for i in I),
    #                         f"Reserv_Virtual_Constraint_tp{j}_{t}",
    #                     )
    #                 else:
    #                     prob += (
    #                         rv[j][t]
    #                         == rv[j][t - 1]
    #                         + W[(j, t)] * y[j][t]
    #                         - pulp.lpSum(d[j][i][t] for i in I),
    #                         f"Reserv_Virtual_Constraint_tp{j}_{t}",
    #                     )
    #             elif j in CF:
    #                 if t == 0:
    #                     prob += (
    #                         rv[j][t]
    #                         == S0 + f[j][t] - pulp.lpSum(d[j][i][t] for i in I),
    #                         f"Reserv_Virtual_Constraint_ff{j}_{t}",
    #                     )
    #                 else:
    #                     prob += (
    #                         rv[j][t]
    #                         == rv[j][t - 1]
    #                         + f[j][t]
    #                         - pulp.lpSum(d[j][i][t] for i in I),
    #                         f"Reserv_Virtual_Constraint_ff{j}_{t}",
    #                     )

    # constr_reserv_virtual(prob)

    # Demanda
    # def constr_demanda(prob):
    #     for t in T:
    #         for i in I:
    #             prob += (
    #                 pulp.lpSum(d[j][i][t] for j in J) == D[(i, t)],
    #                 f"Demanda_Constraint_{i}_{t}",
    #             )

    # constr_demanda(prob)

    # # Carga travada
    # def constr_carga_travada(prob, Y_LOCKED, N_CEN):
    #     for (j, t), valor in Y_LOCKED.items():
    #         if N_CEN - t < 8:
    #             K = N_CEN - t
    #         else:
    #             K = 8
    #         prob += (
    #             k[j][t] >= valor - pulp.lpSum(y[j][t + k] for k in range(K)),
    #             f"Carga_Travada_Constraint_1_{j}_{t}",
    #         )
    #         prob += (
    #             k[j][t] >= pulp.lpSum(y[j][t + k] for k in range(K)) - valor,
    #             f"Carga_Travada_Constraint_2_{j}_{t}",
    #         )

    # constr_carga_travada(prob, Y_LOCKED, N_CEN)

    # Spot
    def constr_spot_1(prob, T, Q2):
        for t in T:
            prob += x[t] <= Q2 * y[len(J) - 1][t], f"Spot_Constraint_{t}"

    constr_spot_1(prob, T, Q2)

    # def constr_spot_top_1(prob, T, Q1, Q2):
    #     for t in T:
    #         prob += (
    #             x[t] - Q1 * y[len(J) - 1][t] <= Q2 * (1 - z[t]),
    #             f"Spot_Top_1_Constraint_{t}",
    #         )

    # def constr_spot_top_2(prob, T, Q1, Q2):
    #     for t in T:
    #         prob += (
    #             Q1 * y[len(J) - 1][t] - x[t] <= Q2 * z[t],
    #             f"Spot_Top_2_Constraint_{t}",
    #         )

    # constr_spot_top_1(prob, T, Q1, Q2)
    # constr_spot_top_2(prob, T, Q1, Q2)

    # Inventário IDLE
    def constr_store_capacity_L_idle(prob, T, S_MIN_IDLE):
        for t in T:
            prob += (
                s[t] >= S_MIN_IDLE * IDLE[t],
                f"Store_Capacity_L_Idle_Constraint_{t}",
            )

    constr_store_capacity_L_idle(prob, T, S_MIN_IDLE)

    # Inventário Prod
    def constr_store_capacity_L_prod(prob, T, S_MIN_PROD, IDLE, s, s_min, SMIN_FLEX):
        for t in T:
            if SMIN_FLEX:
                prob += (
                    s[t] >= S_MIN_PROD * (1 - IDLE[t]) - s_min[t],
                    f"Store_Capacity_L_Prod_Constraint_{t}",
                )
            else:
                prob += (
                    s[t] >= S_MIN_PROD * (1 - IDLE[t]),
                    f"Store_Capacity_L_Prod_Constraint_{t}",
                )

    constr_store_capacity_L_prod(prob, T, S_MIN_PROD, IDLE, s, s_min, SMIN_FLEX)

    # Intervalos de carga
    def constr_interval(prob, T, J, y):
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

    constr_interval(prob, T, J, y)

    # # Tipo de modelo
    # if 1 in IDLE and BOGCTE == False:
    #     for t in T:
    #         if IDLE[t] == 1:
    #             for n in N:
    #                 prob += VL[n] * by[t][n] - s[t] <= 0, f"Boglevel_Lower_{t}_{n}"
    #                 prob += (
    #                     VH[n] * by[t][n] + S_MAX * (1 - by[t][n]) - s[t] >= 0,
    #                     f"Boglevel_Upper_{t}_{n}",
    #                 )
    #             prob += sum(by[t][n] for n in N) == 1, f"Boglevel_Sum_{t}"
    #             prob += b[t] == sum(BI[n] * by[t][n] for n in N), f"Boglevel_b_{t}"
    #         else:
    #             prob += b[t] == BP, f"Boglevel_b_fixed_{t}"
    #             prob += sum(by[t][n] for n in N) == 0, f"Boglevel_by_fixed_{t}"

    # else:
    #     for t in T:
    #         if IDLE[t] == 1:
    #             prob += b[t] == BG, f"BogConstante_b_{t}"
    #         else:
    #             prob += b[t] == BP, f"BogConstante_b_fixed_{t}"

    # Demurrage/ Uma carga por intervalo
    constr_demurrage_interval = []
    for j in TP:
        for interval in INT[j]:
            constr_demurrage_interval.append(
                prob.addConstraint(
                    pulp.lpSum(y[j][t] for t in range(interval[0], interval[1])) <= 1,
                    f"Demurrage_Interval_Constraint_{j}_{interval[0]}_{interval[1]}",
                )
            )

    # # Carga exportação
    # constr_export_cargo = []

    # for j in range(len(J) - 1):
    #     for t in T:
    #         if Y_EXPORT[(j, t)] == 1:
    #             constr_export_cargo.append(
    #                 prob.addConstraint(y[j][t] == 1, f"Export_Cargo_Constraint_{j}_{t}")
    #             )
    # for j in range(len(J) - 1):
    #     for t in T:
    #         if Y_EXPORT[(j, t)] == 1:
    #             constr_export_cargo.append(
    #                 prob.addConstraint(y[j][t] == 1, f"Export_Cargo_Constraint_{j}_{t}")
    #             )

    prob.solve(solver)
    return prob


    def extrair_resultados(self):
        T = self.number_index
        I = range(self.params_modelo["NR_QUANTIDADE_DE_DEMANDAS"])
        J = range(self.params_modelo["NR_QUANTIDADE_DE_CONTRATOS"])
        N = range(7)
        M = range(self.params_modelo["NR_MAX_CATEGORIAS_TOP"])
        C = range(self.params_modelo["NR_MAX_CATEGORIAS_LNGCOST"])
        TP = self.params_modelo["NR_LIST_CONJ_CONTRATOS_TOP"]
        CF = self.params_modelo["NR_LIST_CONJ_CONTRATOS_FF"]
        J1 = range(self.params_modelo["NR_QUANTIDADE_DE_CONTRATOS"] - 1)
        
        
        

        variables_dict = self.model.variablesDict()

        self.resultado_modelo.update({"status": self.model.sol_status})

        # resultado_modelo.update(
        #     {
        #         "d": otimizador.results_dict_to_list(
        #             variables_dict, var="d", index1=J, index2=I, index3=T
        #         )
        #     }
        # )
        # resultado_modelo.update(
        #     {
        #         "rv": otimizador.results_dict_to_list(
        #             variables_dict, var="rv", index1=J, index2=T
        #         )
        #     }
        # )
        self.resultado_modelo.update(
            {"x": otimizador.results_dict_to_list(variables_dict, var="x", index1=T)}
        )
        self.resultado_modelo.update(
            {
                "y": otimizador.results_dict_to_list(
                    variables_dict, var="y", index1=J, index2=T
                )
            }
        )
        # self.resultado_modelo.update(
        #     {
        #         "k": otimizador.results_dict_to_list(
        #             variables_dict, var="k", index1=TP, index2=T
        #         )
        #     }
        # )
        self.resultado_modelo.update(
            {"s": otimizador.results_dict_to_list(variables_dict, var="s", index1=T)}
        )
        self.resultado_modelo.update(
            {
                "s_min": otimizador.results_dict_to_list(
                    variables_dict, var="s_min", index1=T
                )
            }
        )
        # resultado_modelo.update(
        #     {"b": otimizador.results_dict_to_list(variables_dict, var="b", index1=T)}
        # )
        # resultado_modelo.update(
        #     {"z": otimizador.results_dict_to_list(variables_dict, var="z", index1=T)}
        # )
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
        self.resultado_modelo.update(
            {
                "f": otimizador.results_dict_to_list(
                    variables_dict, var="f", index1=CF, index2=T
                )
            }
        )
        self.resultado_modelo.update(
            {
                "fy": otimizador.results_dict_to_list(
                    variables_dict, var="fy", index1=CF, index2=C, index3=T
                )
            }
        )

        if (
            1 in self.params_modelo["NR_LIST_STATUS_IDLE"]
            and self.params_cenario["BOGCTE"] == False
        ):
            self.resultado_modelo.update(
                {
                    "by": otimizador.results_dict_to_list(
                        variables_dict, var="by", index1=T, index2=N
                    )
                }
            )

        
        logging.info(f"Extração finalizada. ")
        
        
