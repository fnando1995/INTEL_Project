from network.classes import classes


def cl_set_counter_data(datadict):

    objects_to_detect = sorted(list(datadict.keys()))
    data_saver = {}
    all_relations = []
    for object in objects_to_detect:
        data_saver[object] = {}
        regions_relations_of_object = datadict[object]
        # parameters are the regions_names involved in the relation
        for relation, parameters in regions_relations_of_object.items():
            if relation == "from_to":
                for parameter in parameters:
                    relation_full_name = relation + "__" + parameter[0] + "_" + parameter[1]
                    data_saver[object][relation_full_name] = 0
                    all_relations.append(relation_full_name)
            elif False:
                pass
                "Nota: se pueden ir añadiendo más casos"
            else:
                print("counter: set_counter_data: relation not recognized: {}".format(relation), parameters)

    all_relations = sorted(list(set(all_relations)))
    relations_per_object = {objeto: sorted(list(data_saver[objeto].keys())) for objeto in
                            sorted(list(objects_to_detect))}
    max_rel_long = max([len(i) for i in all_relations])

    return objects_to_detect, data_saver, all_relations, relations_per_object, max_rel_long


def cl_try_to_count(trk, relations_per_object, counter_data):
    validacion_contado = None
    path = trk.get_path()
    objeto = classes[int(trk.objclass)]
    for relation_full_name in relations_per_object[objeto]:
        rel, other = relation_full_name.split("__")
        if rel == "from_to":
            "Aqui se encuentra la logica para medir si esta relacion se cumple"
            "algoritmo v0.1 - mejorable"
            region_X, region_Y = other.split("_")
            for i in range(len(path[:-1])):  # se revisa hasta el path[-1] además por ser indexacion desde cero -1
                if path[i] == region_X and path[i + 1] == region_Y:
                    if not i + 2 == len(path):
                        if region_X != path[i + 2]:
                            print("trk fue contado en relacion {} debido a un ingreso completo".format(relation_full_name))
                            validacion_contado = True
                            counter_data[objeto][relation_full_name] += 1
                            break
                    if i + 1 == len(path) - 1:
                        # si lo medido es lo último puede ser contado sin problemas
                        print("trk fue contado en relacion {} debido a un ingreso completo".format(relation_full_name))
                        validacion_contado = True
                        counter_data[objeto][relation_full_name] += 1
                        break
    return validacion_contado