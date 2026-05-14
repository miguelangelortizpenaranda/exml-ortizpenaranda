from plotting import get_algorithm_label

def convert_results(algorithms, avg_optimal_rates, total_regret):
    """
    Procesa los resultados de decisiones óptimas y crea un conjunto con los mismos
    tomando como claves los nombres de cada algoritmo
    :param algorithms: Listado de algoritmos a incluir
    :param avg_optimal_rates: Decisiones óptimas realizadas para cada algoritmo
    :param total_regret: Rechazo total
    :return: Conjunto de valores transformado
    """
    # Mostrar resultados
    results = {}
    for i, rate in enumerate(avg_optimal_rates):
        algorithm_label = get_algorithm_label(algorithms[i])
        regret = total_regret[i]
        results[algorithm_label] = rate
        print(f"Algoritmo {algorithm_label}: {rate:.2f}% de elecciones óptimas en el 20% último de steps")
        print(f"Algoritmo {algorithm_label}: {regret:.2f} de arrepentimiento acumulado\n")
    return results


def get_results_average(*results):
    """
    Función de apoyo para obtener un promedio entre varios sets de resultados
    :param results: Conjunto variable de sets de resultados
    :return: Set de resultados con valores promediados para cada key (algoritmo)
    """
    keys = results[0].keys()
    return {
        k: sum(r[k] for r in results) / len(results)
        for k in keys
    }