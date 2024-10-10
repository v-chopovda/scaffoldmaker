"""
Generates a hermite x bilinear 1-D central line mesh for a vagus nerve with branches.
"""
import os
import math
import tempfile

from cmlibs.utils.zinc.field import findOrCreateFieldCoordinates, get_group_list, find_or_create_field_group, \
    create_field_group, get_managed_field_names
from cmlibs.zinc.element import Element, Elementbasis, Elementfieldtemplate
from cmlibs.zinc.field import Field, FieldGroup
from cmlibs.zinc.node import Node

from cmlibs.maths.vectorops import add, sub, magnitude, mult, div, dot

from scaffoldmaker.annotation.annotationgroup import AnnotationGroup, findOrCreateAnnotationGroupForTerm, \
    findAnnotationGroupByName
from scaffoldmaker.annotation.vagus_terms import get_vagus_branch_term, get_vagus_marker_term
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base

from scaffoldmaker.utils.interpolation import getCubicHermiteBasis, interpolateCubicHermite, interpolateLagrangeHermite, \
    interpolateHermiteLagrange, sampleCubicHermiteCurves
from scaffoldmaker.utils.zinc_utils import get_nodeset_field_parameters, print_node_field_parameters
from scaffoldmaker.utils.eft_utils import remapEftNodeValueLabelWithNodes
from scaffoldfitter.fitter import Fitter
from scaffoldfitter.fitterstepconfig import FitterStepConfig
from scaffoldfitter.fitterstepalign import FitterStepAlign
from scaffoldfitter.fitterstepfit import FitterStepFit

from scaffoldmaker.utils.read_vagus_data import load_vagus_data


class MeshType_3d_vagus1(Scaffold_base):
    """
    Generates a hermite x bilinear 1-D central line mesh for a vagus nerve with branches.
    """

    @staticmethod
    def getName():
        return "3D Vagus 1"

    @staticmethod
    def getParameterSetNames():
        return [
            'Human Trunk 1']

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        options = {
            'Number of elements along the trunk': 30,
            'Iterations (fit trunk)': 1,
            'Apply fitting': False,
            'Apply fitting to trunk only': False,
            'Add branches': True
        }
        return options

    @staticmethod
    def getOrderedOptionNames():
        return [
            'Number of elements along the trunk',
            'Iterations (fit trunk)',
            'Apply fitting',
            'Apply fitting to trunk only',
            'Add branches'
        ]

    @classmethod
    def checkOptions(cls, options):
        dependentChanges = False
        if options['Number of elements along the trunk'] < 10:
            options['Number of elements along the trunk'] = 10
        if options['Iterations (fit trunk)'] < 1:
            options['Iterations (fit trunk)'] = 1

        if options['Apply fitting']:
            options['Apply fitting to trunk only'] = False
        if options['Apply fitting to trunk only']:
            options['Apply fitting'] = False

        return dependentChanges


    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the base hermite-bilinear mesh. See also generateMesh().
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: list of AnnotationGroup, None
        """

        # setup
        fieldmodule = region.getFieldmodule()
        fieldcache = fieldmodule.createFieldcache()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

        value_labels = [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1]

        # geometric coordinates
        coordinates = findOrCreateFieldCoordinates(fieldmodule).castFiniteElement()
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        for value_label in value_labels:
            nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)

        mesh = fieldmodule.findMeshByDimension(1)
        elementbasis = fieldmodule.createElementbasis(1, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
        eft = mesh.createElementfieldtemplate(elementbasis)
        elementtemplate = mesh.createElementtemplate()
        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_LINE)
        elementtemplate.defineField(coordinates, -1, eft)

        # branch special node and element field template - geometric coordinates
        nodeTemplateNoValue = nodes.createNodetemplate()
        nodeTemplateNoValue.defineField(coordinates)
        nodeTemplateNoValue.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 0)
        nodeTemplateNoValue.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)

        cubicHermiteBasis = fieldmodule.createElementbasis(1, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
        eftNV = mesh.createElementfieldtemplate(cubicHermiteBasis)
        eftNV.setNumberOfLocalNodes(4)
        eftNV.setNumberOfLocalScaleFactors(4)
        for i in range(4):
            eftNV.setScaleFactorType(i + 1, Elementfieldtemplate.SCALE_FACTOR_TYPE_ELEMENT_GENERAL)
        eftNV.setFunctionNumberOfTerms(1, 4)  # 4 terms = 4 cubic basis functions
        eftNV.setTermNodeParameter(1, 1, 3, Node.VALUE_LABEL_VALUE, 1)
        eftNV.setTermScaling(1, 1, [1])
        eftNV.setTermNodeParameter(1, 2, 3, Node.VALUE_LABEL_D_DS1, 1)
        eftNV.setTermScaling(1, 2, [2])
        eftNV.setTermNodeParameter(1, 3, 4, Node.VALUE_LABEL_VALUE, 1)
        eftNV.setTermScaling(1, 3, [3])
        eftNV.setTermNodeParameter(1, 4, 4, Node.VALUE_LABEL_D_DS1, 1)
        eftNV.setTermScaling(1, 4, [4])
        elementtemplateBranchRoot = mesh.createElementtemplate()
        elementtemplateBranchRoot.setElementShapeType(Element.SHAPE_TYPE_LINE)
        elementtemplateBranchRoot.defineField(coordinates, -1, eftNV)

        elementsAlongTrunk = options['Number of elements along the trunk']
        iterationsNumber = options['Iterations (fit trunk)']
        applyFittingToTrunkOnly = options['Apply fitting to trunk only']
        applyFitting = options['Apply fitting']
        addBranches = options['Add branches']

        # load data from file
        print('Extracting data...')

        data_region = region.getParent().findChildByName('data')
        if data_region.isValid():
            marker_data, trunk_group_name, trunk_data, _, branch_data, branch_parents, _, vagus_terms, fitter_data_file = \
                load_vagus_data(data_region)
        assert len(marker_data) >= 2, f"At least two landmarks are expected in the data. Incomplete data."

        # field group used for fitting
        trunkFitCentroidGroup = find_or_create_field_group(fieldmodule, trunk_group_name + '-fit')
        trunkFitCentroidGroup.setSubelementHandlingMode(FieldGroup.SUBELEMENT_HANDLING_MODE_FULL)
        trunkFitCentroidMeshGroup = trunkFitCentroidGroup.getOrCreateMeshGroup(mesh)

        print('Building centerlines for scaffold...')

        annotationGroups = []

        for ii in range(iterationsNumber):
            # annotations
            vagusTrunkGroup = AnnotationGroup(region, get_vagus_branch_term(trunk_group_name, vagus_terms))
            annotationGroups.append(vagusTrunkGroup)
            vagusTrunkMeshGroup = vagusTrunkGroup.getMeshGroup(mesh)

            if ii == 0:
                trunk_data_endpoints = find_dataset_endpoints_optimised([trunk_pt[0] for trunk_pt in trunk_data])
                tx, td1, _,  elementLength = estimate_trunk_coordinates(elementsAlongTrunk, marker_data, trunk_data_endpoints)
            else:
                # read tx from fit_coordinates
                _, node_field_parameters = get_nodeset_field_parameters(nodes, coordinates, value_labels)
                tx = [nodeParameter[1][0][0] for nodeParameter in node_field_parameters]
                td1 = [nodeParameter[1][1][0] for nodeParameter in node_field_parameters]

            trunk_nodes_data_bounds = estimate_trunk_data_boundaries(tx, elementsAlongTrunk, trunk_data_endpoints)
            print(trunk_nodes_data_bounds)

            nodeIdentifier = 1
            elementIdentifier = 1

            nodes_before = []
            nodes_after = []
            for n in range(elementsAlongTrunk):
                lx = tx[n]
                ld1 = td1[n]

                if ii == 0:
                    node = nodes.createNode(nodeIdentifier, nodetemplate)
                else:
                    node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, lx)
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, ld1)

                # add to trunk group used for data fitting
                if trunk_nodes_data_bounds[0] <= nodeIdentifier <= trunk_nodes_data_bounds[-1]:
                    pass
                elif nodeIdentifier < trunk_nodes_data_bounds[0]:
                    nodes_before.append(nodeIdentifier)
                else:
                    nodes_after.append(nodeIdentifier)

                if n > 0:
                    nids = [nodeIdentifier - 1, nodeIdentifier]
                    if ii == 0:
                        line = mesh.createElement(elementIdentifier, elementtemplate)
                    else:
                        line = mesh.findElementByIdentifier(elementIdentifier)
                    line.setNodesByIdentifier(eft, nids)
                    vagusTrunkMeshGroup.addElement(line)
                    # add element to trunk group used for data fitting
                    if nodeIdentifier - 1 >= trunk_nodes_data_bounds[0] and nodeIdentifier <= trunk_nodes_data_bounds[-1]:
                        trunkFitCentroidMeshGroup.addElement(line)
                    elementIdentifier += 1
                nodeIdentifier += 1

            if ii == 0:
                # set markers
                for marker_name, marker_coordinate in marker_data.items():
                    annotationGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                         get_vagus_marker_term(marker_name),
                                                                         isMarker=True)
                    annotationGroup.createMarkerNode(nodeIdentifier, coordinates, marker_coordinate)
                    nodeIdentifier += 1
            else:
                nodeIdentifier += len(marker_data)

            # geometry fitting - trunk
            if applyFitting or applyFittingToTrunkOnly:

                # create temporary model file
                sir = region.createStreaminformationRegion()
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    fitter_model_file = temp_file.name
                    srf = sir.createStreamresourceFile(fitter_model_file)
                    region.write(sir)

                print('... Fitting trunk, iteration', str(ii + 1))
                fitter = fit_trunk_model(fitter_model_file, fitter_data_file, trunk_group_name + '-fit')
                set_fitted_group_nodes(region, fitter, trunk_group_name + '-fit')

                # remove temporary model file
                os.remove(fitter_model_file)

                if len(nodes_before) > 0 or len(nodes_after):
                    # calculate average derivative d1 along the vagus
                    trunk_fit_nodes = trunkFitCentroidGroup.getNodesetGroup(nodes)
                    node_iter = nodes.createNodeiterator()
                    node = node_iter.next()
                    avg_d1 = [0, 0, 0]
                    while node.isValid():
                        fieldcache.setNode(node)
                        _, ld1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
                        avg_d1 = add(avg_d1, ld1)
                        node = node_iter.next()
                    avg_d1 = [dim / trunk_fit_nodes.getSize() for dim in avg_d1]

                    if len(nodes_before) > 0:
                        # start unfitted nodes from the first fitted node coordinate
                        node_id = trunk_nodes_data_bounds[0]
                        node = nodes.findNodeByIdentifier(node_id)
                        fieldcache.setNode(node)
                        _, lx = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)

                        node_count = 1
                        for i in range(len(nodes_before) - 1, -1, -1):
                            node_id = nodes_before[i]
                            x = [lx[j] - node_count * avg_d1[j] for j in range(3)]

                            node = nodes.findNodeByIdentifier(node_id)
                            fieldcache.setNode(node)
                            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, avg_d1)
                            node_count += 1

                    if len(nodes_after) > 0:
                        # start unfitted nodes from the last fitted node coordinate
                        node_id = trunk_nodes_data_bounds[-1]
                        node = nodes.findNodeByIdentifier(node_id)
                        fieldcache.setNode(node)
                        _, lx = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)

                        node_count = 1
                        for i in range(len(nodes_after)):
                            node_id = nodes_after[i]
                            x = [lx[j] + node_count * avg_d1[j] for j in range(3)]

                            node = nodes.findNodeByIdentifier(node_id)
                            fieldcache.setNode(node)
                            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, avg_d1)
                            node_count += 1

        if addBranches:
            visited_branches = []

            print('... Adding branches')
            queue = [branch for branch in branch_parents.keys() if branch_parents[branch] == trunk_group_name]
            while queue:
                branch_name = queue.pop(0)
                print(branch_name)

                if branch_name in visited_branches:
                    continue
                visited_branches.append(branch_name)
                queue.extend([branch for branch in branch_parents.keys() if branch_parents[branch] == branch_name])

                branch_coordinates = [branch_node[0] for branch_node in branch_data[branch_name]]
                branch_parent_name = branch_parents[branch_name]
                #print(branch_name, ' -> ', branch_parent_name)

                # determine branch approximate start and closest trunk node index
                bx, bd1, parent_s_nid, parent_f_nid, branch_root_xi, elementsAlongBranch = \
                    estimate_branch_coordinates(region, branch_coordinates, elementLength, branch_parent_name)
                #print('  branch between nodes:', parent_s_nid, parent_f_nid, 'at loc =', branch_root_xi)

                branchGroup = AnnotationGroup(region, (branch_name, 'None'))
                annotationGroups.append(branchGroup)
                branchMeshGroup = branchGroup.getMeshGroup(mesh)

                for n in range(elementsAlongBranch):
                    sx = bx[n]
                    sd1 = bd1

                    if n == 0:
                        # create branch special node
                        node = nodes.createNode(nodeIdentifier, nodeTemplateNoValue)
                        fieldcache.setNode(node)
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, sd1)
                    else:
                        node = nodes.createNode(nodeIdentifier, nodetemplate)
                        fieldcache.setNode(node)
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, sx)
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, sd1)

                        if n == 1:
                            # create branch root element
                            nids = [nodeIdentifier - 1, nodeIdentifier,
                                    parent_s_nid, parent_f_nid]
                            element = mesh.createElement(elementIdentifier, elementtemplateBranchRoot)
                            element.setNodesByIdentifier(eftNV, nids)
                            scalefactorsNV = getCubicHermiteBasis(branch_root_xi)
                            element.setScaleFactors(eftNV, list(scalefactorsNV))
                            branchMeshGroup.addElement(element)
                            elementIdentifier += 1
                        else:
                            nids = [nodeIdentifier - 1, nodeIdentifier]
                            element = mesh.createElement(elementIdentifier, elementtemplate)
                            element.setNodesByIdentifier(eft, nids)
                            branchMeshGroup.addElement(element)
                            elementIdentifier += 1
                    nodeIdentifier += 1

                # remove trunk nodes from branch group
                parentGroup = find_or_create_field_group(fieldmodule, branch_parent_name)
                branchNodesetGroup = branchGroup.getNodesetGroup(nodes)
                if branchNodesetGroup.isValid():
                    branchNodesetGroup.removeNodesConditional(parentGroup)

                # geometry fitting - branches
                if applyFitting:
                    # create temporary model file
                    sir = region.createStreaminformationRegion()
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        fitter_model_file = temp_file.name
                        srf = sir.createStreamresourceFile(fitter_model_file)
                        region.write(sir)

                    #print('fitting %s' % branch_name)
                    fitter = fit_branches_model(fitter_model_file, fitter_data_file, branch_name)
                    set_fitted_group_nodes(region, fitter, branch_name)

                    # remove temporary model file
                    os.remove(fitter_model_file)

        # remove temporary data file
        os.remove(fitter_data_file)

        print('Done\n')

        return annotationGroups, None


# supplementary functions
def magnitude_squared(v):
    '''
    return: squared scalar magnitude of vector v
    '''

    # TODO: proposed function to cmlibs.maths
    return sum(c * c for c in v)

def distance_squared(u, v):
    '''
    return: squared distance to avoid the cost of a square root
    '''

    # TODO: proposed function to cmlibs.maths
    return sum((u_i - v_i) ** 2 for u_i, v_i in zip(u, v))

def find_dataset_endpoints(coordinate_dataset):
    """
    Given list of XYZ coordinates, find two furthest apart from each other.
    Returns a list of two coordinates.
    """

    max_distance_squared = 0
    ends_points = []
    for node_x in coordinate_dataset:
        for node_y in coordinate_dataset:
            dist = distance_squared(node_x, node_y)
            if dist >= max_distance_squared:
                max_distance_squared = dist
                ends_points = [node_y, node_x]
    return ends_points

def find_dataset_endpoints_optimised(points):
    """
        Given list of XYZ coordinates, find two furthest apart from each other.
        Returns a list of two coordinates.
    """
    if len(points) < 2:
        raise ValueError("find_dataset_endpoints: At least two points are required.")

    # Step 1: Deterministically select points for the sample (every step_size-th point)
    step_size = 1 if len(points) < 200 else 200
    sampled_indices = list(range(0, len(points), step_size))
    if len(points) - 1 not in sampled_indices:
        sampled_indices.append(len(points) - 1)

    # Step 2: Find the furthest points from within the sampled subset
    max_distance = 0
    furthest_pair = (sampled_indices[0], sampled_indices[1])

    for i in range(len(sampled_indices)):
        for j in range(i + 1, len(sampled_indices)):
            dist = distance_squared(points[sampled_indices[i]], points[sampled_indices[j]])
            if dist > max_distance:
                max_distance = dist
                furthest_pair = (sampled_indices[i], sampled_indices[j])

    # Step 3: Compare sampled endpoints against all points, maintaining original order
    for i in range(len(points)):
        for endpoint_index in furthest_pair:
            if i < endpoint_index:
                dist = distance_squared(points[i], points[endpoint_index])
                if dist > max_distance:
                    max_distance = dist
                    furthest_pair = (i, endpoint_index)

    endpoints = [points[furthest_pair[0]], points[furthest_pair[1]]]
    return endpoints


def find_point_projection_relative_to_segment(point, segment_start, segment_end):
    """
    return: the position of the projection relative to the line segment, defined by segment_start and segment_end
    """

    ap = sub(point, segment_start)
    ab = sub(segment_end, segment_start)
    projection_scalar = dot(ap, ab) / dot(ab, ab) if dot(ab, ab) != 0 else 0
    return projection_scalar

def estimate_total_vagus_length(marker_data):

    # check if any data below esophageal plexus is supplied
    if any(['level of esophageal hiatus' in marker_name or 'level of aortic hiatus' in marker_name for marker_name in
            marker_data.keys()]):

        totalVagusLength = 312.5  # calculated from total length of nerve/average diameter of nerve
    else:
        totalVagusLength = 166.25

    return totalVagusLength


def estimate_trunk_coordinates(elementsAlongTrunk, marker_data, trunk_data_endpoints):
    """

    """

    # choose markers for building initial scaffold
    # at the moment uses the first and the last markers in the data
    termNameVagusLengthList = {
        # cervical region
        "level of exiting brainstem on the vagus nerve": 0.0,  # note this term is not on the list of annotations
        "level of superior border of jugular foramen on the vagus nerve": 8.6342,
        "level of inferior border of jugular foramen on the vagus nerve": 16.7227,
        "level of inferior border of cranium on the vagus nerve": 18.375,
        "level of C1 transverse process on the vagus nerve": 32.1129,
        "level of angle of mandible on the vagus nerve": 42.2450,
        "level of greater horn of hyoid on the vagus nerve": 45.6122,
        "level of carotid bifurcation on the vagus nerve": 48.3581,
        "level of laryngeal prominence on the vagus nerve": 68.8431,
        # thoracic region
        "level of superior border of the clavicle on the vagus nerve": 117.5627,
        "level of jugular notch on the vagus nerve": 124.6407,
        "level of carina": 149.5929,  # not on the list of annotations yet!
        "level of sternal angle on the vagus nerve": 151.2352,
        "level of 1 cm superior to start of esophageal plexus on the vagus nerve": 165.5876,
        # abdominal region
        "level of esophageal hiatus on the vagus nerve": 254.32879,
        "level of aortic hiatus on the vagus nerve": 291.3695,
        "level of end of trunk": 312.5  # note this term is also not on the list of annotations
    }

    totalVagusLength = estimate_total_vagus_length(marker_data)
    if totalVagusLength == termNameVagusLengthList["level of end of trunk"]:
        print('Estimate_trunk: Vagus top to bottom')
    else:
        print('Estimate_trunk: Vagus top to esophageal plexus')

    step = totalVagusLength / (elementsAlongTrunk - 1)
    elementLength = magnitude(sub(trunk_data_endpoints[0], trunk_data_endpoints[-1])) / (elementsAlongTrunk - 1)

    use_markers = [list(marker_data.keys())[0],
                   list(marker_data.keys())[-1]]

    pts = []
    params = []

    for marker in use_markers:
        use_marker_name = marker.replace('left ', '', 1).replace('right ', '', 1)
        assert use_marker_name in termNameVagusLengthList

        pts.append(marker_data[marker])
        params.append(termNameVagusLengthList[use_marker_name])
    dx, dy, dz = [(pts[1][dim] - pts[0][dim]) / (params[1] - params[0]) for dim in range(3)]

    trunk_nodes = []
    trunk_d1 = []
    for i in range(elementsAlongTrunk):
        trunk_nodes.append([pts[0][0] + dx * (i * step - params[0]),
                            pts[0][1] + dy * (i * step - params[0]),
                            pts[0][2] + dz * (i * step - params[0])])
        trunk_d1.append([dx * step, dy * step, dz * step])

    return trunk_nodes, trunk_d1, step, elementLength


def estimate_trunk_data_boundaries(trunk_nodes, elementsAlongTrunk, trunk_data_endpoints):
    """
    """

    trunk_nodes_data_bounds = []
    for ind, endpoint in enumerate(trunk_data_endpoints):
        param = find_point_projection_relative_to_segment(endpoint, trunk_nodes[0], trunk_nodes[-1])
        if param < 0:
            nearby_node = 1
        elif param > 1:
            nearby_node = elementsAlongTrunk
        else:
            nearby_node = param * (elementsAlongTrunk - 1) + 1

        if ind == 0:
            # trunk node near the start of data
            trunk_nodes_data_bounds.append(math.floor(nearby_node))
        else:
            # trunk node near the end of data
            trunk_nodes_data_bounds.append(math.ceil(nearby_node))

    return trunk_nodes_data_bounds


def estimate_branch_coordinates(region, branch_coordinates, elementLength, branch_parent_name):
    """

    """

    fm = region.getFieldmodule()
    fieldcache = fm.createFieldcache()
    coordinates = fm.findFieldByName("coordinates").castFiniteElement()
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

    # assumes parent_group_name is known
    branch_start_x, branch_end_x, parent_s_nid, parent_f_nid = find_branch_start_segment(region, branch_coordinates,
                                                                                         branch_parent_name)

    # determine parent hermite curve parameters
    node = nodes.findNodeByIdentifier(parent_s_nid)
    fieldcache.setNode(node)
    _, px_1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
    _, pd1_1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)

    node = nodes.findNodeByIdentifier(parent_f_nid)
    fieldcache.setNode(node)
    _, px_2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
    _, pd1_2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)

    # find xi closest to branch_start on a cubic Hermite curve by bisection
    xi_a = 0
    xi_b = 1
    eps = 0.005
    while (xi_b - xi_a) > eps:
        dsq_a = distance_squared(branch_start_x, interpolateCubicHermite(px_1, pd1_1, px_2, pd1_2, xi_a))
        dsq_b = distance_squared(branch_start_x, interpolateCubicHermite(px_1, pd1_1, px_2, pd1_2, xi_b))
        if dsq_a >= dsq_b:
            xi_a = (xi_a + xi_b) / 2
        else:
            xi_b = (xi_a + xi_b) / 2
    branch_root_xi = (xi_a + xi_b) / 2

    # recalculate branch start parameters
    branch_start_x = interpolateHermiteLagrange(px_1, pd1_1, px_2, branch_root_xi)
    branch_length = magnitude(sub(branch_end_x, branch_start_x))
    #print('  branch_length,', branch_length)
    elementsAlongBranch = math.floor(branch_length / elementLength) + 1
    if elementsAlongBranch < 3:
        elementsAlongBranch = 3
    if elementsAlongBranch > 10:
        elementsAlongBranch = 10

    branch_coordinates = []
    dx, dy, dz = div(sub(branch_end_x, branch_start_x), (elementsAlongBranch - 1))
    for i in range(elementsAlongBranch):
        branch_coordinates.append([branch_start_x[0] + dx * i,
                                   branch_start_x[1] + dy * i,
                                   branch_start_x[2] + dz * i])

    return branch_coordinates, [dx, dy, dz], parent_s_nid, parent_f_nid, branch_root_xi, elementsAlongBranch


def find_branch_start_segment(region, branch_coordinates, parent_group_name):
    """

    """

    fm = region.getFieldmodule()
    coordinates = fm.findFieldByName("coordinates").castFiniteElement()
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

    parent_group = find_or_create_field_group(fm, parent_group_name)
    parent_nodeset = parent_group.getNodesetGroup(nodes)
    _, parent_group_parameters = get_nodeset_field_parameters(parent_nodeset, coordinates, [Node.VALUE_LABEL_VALUE])
    parent_nodeset_ids = [parameter[0] for parameter in parent_group_parameters]
    parent_nodeset_x = [parameter[1][0][0] for parameter in parent_group_parameters]

    # find branch ends in data
    branch_ends_points = find_dataset_endpoints_optimised(branch_coordinates)

    # find parent node index closest to branch and point where branch starts
    min_dsq = float('inf')
    for i in range(len(parent_nodeset_x)):
        node_x = parent_nodeset_x[i]
        if node_x is None:
            continue

        for branch_point in branch_ends_points:
            dist = distance_squared(node_x, branch_point)
            if dist <= min_dsq:
                min_dsq = dist
                branch_start = branch_point
                closest_index = i

    # determine segment closest to branch (previous or next to the node)
    if closest_index == 0:
        parent_start_index = closest_index
    elif closest_index == len(parent_nodeset_x) - 1:
        parent_start_index = closest_index - 1
    else:
        proj_before = find_point_projection_relative_to_segment(branch_start,
                                                                parent_nodeset_x[closest_index - 1],
                                                                parent_nodeset_x[closest_index])
        proj_after = find_point_projection_relative_to_segment(branch_start,
                                                               parent_nodeset_x[closest_index],
                                                               parent_nodeset_x[closest_index + 1])
        if 0 <= proj_before <= 1:
            parent_start_index = closest_index - 1
        elif 0 <= proj_after <= 1:
            parent_start_index = closest_index
        elif abs(proj_before) < abs(proj_after):
            parent_start_index = closest_index - 1
        else:
            parent_start_index = closest_index

    parent_s_node_id = parent_nodeset_ids[parent_start_index]
    parent_f_node_id = parent_nodeset_ids[parent_start_index + 1]
    branch_end = branch_ends_points[0] if branch_ends_points[1] == branch_start else branch_ends_points[1]

    return branch_start, branch_end, parent_s_node_id, parent_f_node_id


def fit_trunk_model(modelfile, datafile, trunk_group_name = None):
    """

    """
    fitter = Fitter(modelfile, datafile)
    fitter.load()

    # initial configuration
    fitter_fieldmodule = fitter.getFieldmodule()
    fitter.setModelCoordinatesFieldByName('coordinates')
    fitter.setDataCoordinatesFieldByName('coordinates')
    if trunk_group_name:
        fitter.setModelFitGroupByName(trunk_group_name)
    fitter.setFibreField(fitter_fieldmodule.findFieldByName("zero fibres"))
    fitter.setMarkerGroupByName('marker')  # not necessary, it's marker by default
    fitter.setDiagnosticLevel(0)

    # initial fit step in case some data is not covered
    fit0 = FitterStepFit()
    fit0.setGroupDataStretch(None, True)
    fit0.setGroupStrainPenalty(None, [1000.0])
    fit0.setGroupCurvaturePenalty(None, [2000.0])
    fit0.setNumberOfIterations(5)
    fit0.setUpdateReferenceState(False)
    fitter.addFitterStep(fit0)

    # fit step 1
    fit1 = FitterStepFit()
    fit1.setGroupDataWeight('marker', 100.0)
    fit1.setGroupDataSlidingFactor('marker', 0.01)
    fit1.setGroupStrainPenalty(None, [15.0])
    fit1.setGroupCurvaturePenalty(None, [50.0])
    fit1.setNumberOfIterations(10)
    fit1.setUpdateReferenceState(True)
    fitter.addFitterStep(fit1)

    # fit step 2
    fit2 = FitterStepFit()
    fit2.setGroupDataWeight('marker', 100.0)
    fit2.setGroupDataSlidingFactor('marker', 0.01)
    fit2.setGroupStrainPenalty(None, [5.0])
    fit2.setGroupCurvaturePenalty(None, [100.0])
    fit2.setNumberOfIterations(5)
    fit2.setUpdateReferenceState(True)
    fitter.addFitterStep(fit2)

    fitter.run()

    rmsError, maxError = fitter.getDataRMSAndMaximumProjectionError()
    rmsTrunkError, maxTrunkError = fitter.getDataRMSAndMaximumProjectionErrorForGroup('left vagus X nerve trunk')
    rmsMarkerError, maxMarkerError = fitter.getDataRMSAndMaximumProjectionErrorForGroup('marker')

    # print('(all) RMS error: ' + str(rmsError))
    # print('(all) Max error: ' + str(maxError))
    # print('(trunk) RMS error: ' + str(rmsTrunkError))
    # print('(trunk) Max error: ' + str(maxTrunkError))
    # print('(marker) RMS error: ' + str(rmsMarkerError))
    # print('(marker) Max error: ' + str(maxMarkerError))

    # fitter_nodes = fitter_fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    # fitter_coordinates = fitter.getModelCoordinatesField().castFiniteElement()
    # valueLabels, fieldParameters = get_nodeset_field_parameters(fitter_nodes, fitter_coordinates)
    # print_node_field_parameters(valueLabels, fieldParameters, '{: 1.4f}')

    return fitter


def fit_branches_model(modelfile, datafile, branch_name = None):
    """

    """

    # initial configuration
    fitter = Fitter(modelfile, datafile)
    fitter.load()
    fitter.setModelCoordinatesFieldByName('coordinates')
    if branch_name:
        fitter.setModelFitGroupByName(branch_name)
    fitter.setFibreField(fitter.getFieldmodule().findFieldByName("zero fibres"))
    fitter.setDataCoordinatesFieldByName('coordinates')
    fitter.setMarkerGroupByName('marker')  # not necessary, it's marker by default
    fitter.setDiagnosticLevel(0)

    # fit step 1
    fit1 = FitterStepFit()
    fit1.setGroupStrainPenalty(None, [15.0])
    fit1.setGroupCurvaturePenalty(None, [50.0])
    fit1.setNumberOfIterations(5)
    fit1.setUpdateReferenceState(True)
    fitter.addFitterStep(fit1)
    fitter.run()

    rmsError, _ = fitter.getDataRMSAndMaximumProjectionErrorForGroup(branch_name)
    print(branch_name, rmsError)

    if rmsError > 10.0:
        # initial configuration
        fitter2 = Fitter(modelfile, datafile)
        fitter2.load()
        fitter2.setModelCoordinatesFieldByName('coordinates')
        if branch_name:
            fitter.setModelFitGroupByName(branch_name)
        fitter2.setFibreField(fitter2.getFieldmodule().findFieldByName("zero fibres"))
        fitter2.setDataCoordinatesFieldByName('coordinates')
        fitter2.setMarkerGroupByName('marker')  # not necessary, it's marker by default
        fitter2.setDiagnosticLevel(0)

        # fit step 1
        fit2 = FitterStepFit()
        fit2.setGroupDataSlidingFactor(None, 0.1)
        fit2.setGroupStrainPenalty(None, [25.0])
        fit2.setGroupCurvaturePenalty(None, [150.0])
        fit2.setNumberOfIterations(10)
        fitter2.addFitterStep(fit2)
        fitter2.run()

        # additional fit step 3
        fitter3 = fitter2
        fit3 = FitterStepFit()
        fit3.setGroupDataSlidingFactor(None, 0.0001)
        fit3.setGroupStrainPenalty(None, [5.0])
        fit3.setGroupCurvaturePenalty(None, [100.0])
        fit3.setNumberOfIterations(10)
        fitter3.addFitterStep(fit3)
        fitter3.run()

        rmsError2, _ = fitter2.getDataRMSAndMaximumProjectionErrorForGroup(branch_name)
        rmsError3, _ = fitter3.getDataRMSAndMaximumProjectionErrorForGroup(branch_name)
        print('fitter rerun for', branch_name, rmsError, rmsError2, rmsError3)

        if rmsError3 < rmsError2:
            fitter2 = fitter3

        if rmsError2 < rmsError:
            fitter = fitter2

    return fitter


def set_fitted_group_nodes(region, fitter, group_name = None):
    """
    """

    fieldmodule = region.getFieldmodule()
    fieldcache = fieldmodule.createFieldcache()
    coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

    fitter_fieldmodule = fitter.getFieldmodule()
    fitter_fieldcache = fitter_fieldmodule.createFieldcache()
    fitter_coordinates = fitter.getModelCoordinatesField().castFiniteElement()
    fitter_nodes = fitter_fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

    if group_name:
        group = find_or_create_field_group(fitter_fieldmodule, group_name)
        if group.isValid():
            fitter_nodes = group.getNodesetGroup(fitter_nodes)

    # reset trunk nodes with the fitted nodes
    fitter_node_iter = fitter_nodes.createNodeiterator()
    fitter_node = fitter_node_iter.next()
    while fitter_node.isValid():
        fitter_fieldcache.setNode(fitter_node)
        _, lx = fitter_coordinates.getNodeParameters(fitter_fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
        _, ld1 = fitter_coordinates.getNodeParameters(fitter_fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)

        node = nodes.findNodeByIdentifier(fitter_node.getIdentifier())
        fieldcache.setNode(node)
        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, lx)
        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, ld1)
        fitter_node = fitter_node_iter.next()

