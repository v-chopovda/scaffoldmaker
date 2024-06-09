"""
Generates a 3-D Hermite bifurcating tube network.
"""
from scaffoldmaker.meshtypes.meshtype_1d_network_layout1 import MeshType_1d_network_layout1
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.scaffoldpackage import ScaffoldPackage
from scaffoldmaker.utils.tubenetworkmesh import TubeNetworkMeshBuilder, TubeNetworkMeshGenerateData


class MeshType_3d_tubenetwork1(Scaffold_base):
    """
    Generates a 3-D hermite tube network, with linear or cubic basis through wall.
    """

    @classmethod
    def getName(cls):
        return "3D Tube Network 1"

    @classmethod
    def getParameterSetNames(cls):
        return MeshType_1d_network_layout1.getParameterSetNames()

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        options = {
            "Network layout": ScaffoldPackage(MeshType_1d_network_layout1,
                                              {"scaffoldSettings": {"Define inner coordinates": True}},
                                              defaultParameterSetName=parameterSetName),
            "Elements count around": 8,
            "Elements count through wall": 1,
            "Annotation elements counts around": [0],
            "Target element density along longest segment": 4.0,
            "Linear through wall": False,
            "Show trim surfaces": False
        }
        return options

    @staticmethod
    def getOrderedOptionNames():
        return [
            "Network layout",
            "Elements count around",
            "Elements count through wall",
            "Annotation elements counts around",
            "Target element density along longest segment",
            "Linear through wall",
            "Show trim surfaces"
        ]

    @classmethod
    def getOptionValidScaffoldTypes(cls, optionName):
        if optionName == "Network layout":
            return [MeshType_1d_network_layout1]
        return []

    @classmethod
    def getOptionScaffoldTypeParameterSetNames(cls, optionName, scaffoldType):
        assert scaffoldType in cls.getOptionValidScaffoldTypes(optionName), \
            cls.__name__ + ".getOptionScaffoldTypeParameterSetNames.  " + \
            "Invalid option \"" + optionName + "\" scaffold type " + scaffoldType.getName()
        return scaffoldType.getParameterSetNames()  # use the defaults from the network layout scaffold

    @classmethod
    def getOptionScaffoldPackage(cls, optionName, scaffoldType, parameterSetName=None):
        """
        :param parameterSetName:  Name of valid parameter set for option Scaffold, or None for default.
        :return: ScaffoldPackage.
        """
        if parameterSetName:
            assert parameterSetName in cls.getOptionScaffoldTypeParameterSetNames(optionName, scaffoldType), \
                "Invalid parameter set " + str(parameterSetName) + " for scaffold " + str(scaffoldType.getName()) + \
                " in option " + str(optionName) + " of scaffold " + cls.getName()
        if optionName == "Network layout":
            if not parameterSetName:
                parameterSetName = "Default"
            return ScaffoldPackage(scaffoldType,
                                   {"scaffoldSettings": {"Define inner coordinates": True}},
                                   defaultParameterSetName=parameterSetName)
        assert False, cls.__name__ + ".getOptionScaffoldPackage:  Option " + optionName + " is not a scaffold"

    @classmethod
    def checkOptions(cls, options):
        if not options["Network layout"].getScaffoldType() in cls.getOptionValidScaffoldTypes("Network layout"):
            options["Network layout"] = cls.getOptionScaffoldPackage("Network layout", MeshType_1d_network_layout1)
        elementsCountAround = options["Elements count around"]
        if options["Elements count around"] < 4:
            options["Elements count around"] = 4
        if options["Elements count through wall"] < 1:
            options["Elements count through wall"] = 1
        annotationElementsCountsAround = options["Annotation elements counts around"]
        if len(annotationElementsCountsAround) == 0:
            options["Annotation elements count around"] = [0]
        else:
            for i in range(len(annotationElementsCountsAround)):
                if annotationElementsCountsAround[i] <= 0:
                    annotationElementsCountsAround[i] = 0
                elif annotationElementsCountsAround[i] < 4:
                    annotationElementsCountsAround[i] = 4
        if options["Target element density along longest segment"] < 1.0:
            options["Target element density along longest segment"] = 1.0
        dependentChanges = False
        return dependentChanges

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the base tricubic hermite or bicubic hermite-linear mesh. See also generateMesh().
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: list of AnnotationGroup, None
        """
        layoutRegion = region.createRegion()
        networkLayout = options["Network layout"]
        networkLayout.generate(layoutRegion)  # ask scaffold to generate to get user-edited parameters
        networkMesh = networkLayout.getConstructionObject()

        tubeNetworkMeshBuilder = TubeNetworkMeshBuilder(
            networkMesh,
            targetElementDensityAlongLongestSegment=options["Target element density along longest segment"],
            defaultElementsCountAround=options["Elements count around"],
            elementsCountThroughWall=options["Elements count through wall"],
            layoutAnnotationGroups=networkLayout.getAnnotationGroups(),
            annotationElementsCountsAround=options["Annotation elements counts around"])
        tubeNetworkMeshBuilder.build()
        generateData = TubeNetworkMeshGenerateData(
            region, 3,
            isLinearThroughWall=options["Linear through wall"],
            isShowTrimSurfaces=options["Show trim surfaces"])
        tubeNetworkMeshBuilder.generateMesh(generateData)
        annotationGroups = generateData.getAnnotationGroups()

        return annotationGroups, None
