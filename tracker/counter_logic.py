from modules.main_utils import *
from modules.constantes.variables import CLASSES_IN_NETWORK

"""
Estas Logicas estas bien ligadas a los nombres de las regiones que se creen.
Siempre iniciar con A,B,C... etc.
"""


def cl_set_counter_data(datadict):
    """
    Logica de seteo de diccionario que referencia los valores del archivo de figura para
    poder generar los atributos de un contador (Counter())
    Aqui se deben generar (por condicion) que tipo de relacion tiene y como maneja sus
    parametros.
    """
    objects_to_detect = sorted(list(datadict.keys()))
    data_saver = {}
    all_relations = []
    for object in objects_to_detect:
        data_saver[object] = {}
        regions_relations_of_object = datadict[object]
        # parameters are the regions_names involved in the relation
        for relation, parameters in regions_relations_of_object.items():
            if relation == "enters":
                """
                ------------------->enters__A
                Relacion en la que un objeto de None ingresa a la region indicada
                o es inicializada en la region por primera vez
                NOTA: Tener en cuenta que el "inicio" en la region puede aumentar el
                conteo debido a los desperfectos del tracking.
                """
                for parameter in parameters:
                    relation_full_name = relation + "__" + parameter
                    data_saver[object][relation_full_name] = 0
                    all_relations.append(relation_full_name)
                    # print("relation created: {} -> {}".format(object,relation_full_name))
            elif relation == "from_to":
                """
                ------------------->from_to__A_B
                """
                for parameter in parameters:
                    relation_full_name = relation + "__" + parameter[0] + "_" + parameter[1]
                    # print("relation created: {} -> {}".format(object,relation_full_name))
                    data_saver[object][relation_full_name] = 0
                    all_relations.append(relation_full_name)
            else:
                print("counter: set_counter_data: relation not recognized: {}".format(relation), parameters)

    all_relations = sorted(list(set(all_relations)))
    relations_per_object = {objeto: sorted(list(data_saver[objeto].keys())) for objeto in
                            sorted(list(objects_to_detect))}
    max_rel_long = max([len(i) for i in all_relations])

    return objects_to_detect, data_saver, all_relations, relations_per_object, max_rel_long


def cl_try_to_count(trk, relations_per_object, counter_data):
    """
    Esta funcion maneja la logica de conteo para las relaciones establecidas en el sistema.
    """
    validacion_contado = None
    path = trk.get_path()
    # print("Intentanto ver siguiente path:", trk.get_path())
    objeto = CLASSES_IN_NETWORK[int(trk.objclass)]
    for relation_full_name in relations_per_object[objeto]:
        rel, other = relation_full_name.split("__")
        if rel == "from_to":
            """
            Esta relacion debe poder indicar si el trk paso de una region a otra sin hacer retorno.
            Esto se da de la siguiente manera.

            Suponiendo un path = A-B-A-B-A  notamos que hay mucha

            """
            region_X, region_Y = other.split("_")
            for i in range(len(path[:-1])):  # se revisa hasta el path[-1] además por ser indexacion desde cero -1
                if path[i] == region_X and path[i + 1] == region_Y:
                    if not i + 2 == len(path):
                        if region_X != path[i + 2]:
                            reportErrorLogs("trk fue contado en relacion {} debido a un ingreso completo".format(
                                relation_full_name))
                            validacion_contado = True
                            counter_data[objeto][relation_full_name] += 1
                            break
                    if i + 1 == len(path) - 1:
                        # si lo medido es lo último puede ser contado sin problemas
                        reportErrorLogs(
                            "trk fue contado en relacion {} debido a un ingreso completo".format(relation_full_name))
                        validacion_contado = True
                        counter_data[objeto][relation_full_name] += 1
                        break
    return validacion_contado