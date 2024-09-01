"""
Specialisation of Network Mesh for building 2-D and 3-D tube mesh networks.
"""
from cmlibs.maths.vectorops import add, cross, dot, magnitude, mult, normalize, set_magnitude, sub, rejection
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.node import Node
from scaffoldmaker.utils.eft_utils import determineCubicHermiteSerendipityEft, HermiteNodeLayoutManager
from scaffoldmaker.utils.interpolation import (
    computeCubicHermiteDerivativeScaling, DerivativeScalingMode, evaluateCoordinatesOnCurve,
    getCubicHermiteTrimmedCurvesLengths, interpolateCubicHermite, interpolateCubicHermiteDerivative,
    interpolateLagrangeHermiteDerivative, interpolateSampleCubicHermite, sampleCubicHermiteCurves,
    sampleCubicHermiteCurvesSmooth, smoothCubicHermiteDerivativesLine, smoothCubicHermiteDerivativesLoop,
    smoothCurveSideCrossDerivatives)
from scaffoldmaker.utils.networkmesh import NetworkMesh, NetworkMeshBuilder, NetworkMeshGenerateData, \
    NetworkMeshJunction, NetworkMeshSegment, pathValueLabels
from scaffoldmaker.utils.tracksurface import TrackSurface
from scaffoldmaker.utils.zinc_utils import get_nodeset_path_ordered_field_parameters
import copy
import math


class TubeNetworkMeshGenerateData(NetworkMeshGenerateData):
    """
    Data for passing to TubeNetworkMesh generateMesh functions.
    """

    def __init__(self, region, meshDimension, isLinearThroughWall, isShowTrimSurfaces,
            coordinateFieldName="coordinates", startNodeIdentifier=1, startElementIdentifier=1):
        """
        :param isLinearThroughWall: Callers should only set if 3-D with no core.
        :param isShowTrimSurfaces: Tells junction generateMesh to make 2-D trim surfaces.
        """
        super(TubeNetworkMeshGenerateData, self).__init__(
            region, meshDimension, coordinateFieldName, startNodeIdentifier, startElementIdentifier)
        self._isLinearThroughWall = isLinearThroughWall
        self._isShowTrimSurfaces = isShowTrimSurfaces

        # get node template for standard and cross nodes
        self._nodetemplate = self._nodes.createNodetemplate()
        self._nodetemplate.defineField(self._coordinates)
        self._nodetemplate.setValueNumberOfVersions(self._coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        self._nodetemplate.setValueNumberOfVersions(self._coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        if (meshDimension == 3) and not isLinearThroughWall:
            self._nodetemplate.setValueNumberOfVersions(self._coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)

        # get element template and eft for standard case
        self._standardElementtemplate = self._mesh.createElementtemplate()
        self._standardElementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE if (meshDimension == 3)
                                                     else Element.SHAPE_TYPE_SQUARE)
        elementbasis = self._fieldmodule.createElementbasis(
            meshDimension, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY)
        if (meshDimension == 3) and isLinearThroughWall:
            elementbasis.setFunctionType(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        self._standardEft = self._mesh.createElementfieldtemplate(elementbasis)
        self._standardElementtemplate.defineField(self._coordinates, -1, self._standardEft)

        d3Defined = (meshDimension == 3) and not isLinearThroughWall
        self._nodeLayoutManager = HermiteNodeLayoutManager()
        self._nodeLayout6Way = self._nodeLayoutManager.getNodeLayout6Way12(d3Defined)
        self._nodeLayout8Way = self._nodeLayoutManager.getNodeLayout8Way12(d3Defined)
        self._nodeLayoutFlipD2 = self._nodeLayoutManager.getNodeLayoutRegularPermuted(
            d3Defined, limitDirections=[None, [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], [[0.0, 0.0, 1.0]]] if d3Defined
            else [None, [[0.0, 1.0], [0.0, -1.0]]])
        self._nodeLayout6WayTriplePoint = self._nodeLayoutManager.getNodeLayout6WayTriplePoint()
        self._nodeLayoutBifrucation = self._nodeLayoutManager.getNodeLayout6WayBifurcation()
        self._nodeLayoutTrifurcation = None
        self._nodeLayoutTransition = self._nodeLayoutManager.getNodeLayoutRegularPermuted(
            d3Defined, limitDirections=[None, [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], None])
        self._nodeLayoutTransitionTriplePoint = None
        self._nodeLayoutBifurcationTransition = self._nodeLayoutManager.getNodeLayout6WayBifurcationTransition()


    def getStandardEft(self):
        return self._standardEft

    def getStandardElementtemplate(self):
        return self._standardElementtemplate

    def getNodeLayout6Way(self):
        return self._nodeLayout6Way

    def getNodeLayout8Way(self):
        return self._nodeLayout8Way

    def getNodeLayoutFlipD2(self):
        return self._nodeLayoutFlipD2

    def getNodeLayout6WayTriplePoint(self, location):
        """
        Special node layout for generating core transition elements, where a node is located at both 6-way junction
        and one of the triple point corners of core box elements. There are two layouts specific to top and bottom
        corner: Top (location = 1); and bottom right (location = 2).
        :param location: Location identifier.
        :return: Node layout.
        """
        nodeLayouts = self._nodeLayoutManager.getNodeLayout6WayTriplePoint()
        location = abs(location)
        # assert location in [1, 2]

        if location == 1:  # "Top1"
            nodeLayout = nodeLayouts[0]
        elif location == 3:  # "Top1"
            nodeLayout = nodeLayouts[1]
        elif location == 2:  # "Bottom1"
            nodeLayout = nodeLayouts[2]
        elif location == 4:  # "Bottom2"
            nodeLayout = nodeLayouts[3]

        return nodeLayout

    def getNodeLayoutBifurcation(self):
        """
        Special node layout for generating core elements for bifurcation.
        """
        return self._nodeLayoutBifrucation

    def getNodeLayoutTrifurcation(self, location):
        """
        Special node layout for generating core elements for trifurcation. There are two layouts specific to
        left-hand side and right-hand side of the solid core cross-section: LHS (location = 1); and RHS (location = 2).
        :param location: Location identifier.
        :return: Node layout.
        """
        if location == 1:  # Left-hand side
            limitDirections = [None, [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], None]
        elif location == 2:  # Right-hand side
            limitDirections = [None, [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], None]

        self._nodeLayoutTrifurcation = self._nodeLayoutManager.getNodeLayout6Way12(True, limitDirections)
        return self._nodeLayoutTrifurcation

    def getNodeLayoutTransition(self):
        """
        Node layout for generating core transition elements, excluding at triple points.
        """
        return self._nodeLayoutTransition

    def getNodeLayoutBifurcationTransition(self):
        """
        Special node layout for generating core transition elements for bifurcation.
        """
        return self._nodeLayoutBifurcationTransition

    def getNodeLayoutTransitionTriplePoint(self, location):
        """
        Special node layout for generating core transition elements at triple points.
        There are four layouts specific to each corner of the core box: Top left (location = 1);
        top right (location = -1); bottom left (location = 2); and bottom right (location = -2).
        :param location: Location identifier identifying four corners of solid core box.
        :return: Node layout.
        """
        nodeLayouts = self._nodeLayoutManager.getNodeLayoutTriplePoint()
        assert location in [1, -1, 2, -2, 0]
        if location == 1:  # "Top Left"
            nodeLayout = nodeLayouts[0]
        elif location == -1:  # "Top Right"
            nodeLayout = nodeLayouts[1]
        elif location == 2:  # "Bottom Left"
            nodeLayout = nodeLayouts[2]
        elif location == -2:  # "Bottom Right"
            nodeLayout = nodeLayouts[3]
        else:
            nodeLayout = self._nodeLayoutTransition

        self._nodeLayoutTransitionTriplePoint = nodeLayout
        return self._nodeLayoutTransitionTriplePoint

    def getNodetemplate(self):
        return self._nodetemplate

    def isLinearThroughWall(self):
        return self._isLinearThroughWall

    def isShowTrimSurfaces(self):
        return self._isShowTrimSurfaces

class TubeNetworkMeshSegment(NetworkMeshSegment):

    def __init__(self, networkSegment, pathParametersList, elementsCountAround, elementsCountThroughWall,
                 isCore=False, elementsCountAcrossMajor: int = 4, elementsCountAcrossMinor: int = 4,
                 elementsCountTransition: int = 1):
        """
        :param networkSegment: NetworkSegment this is built from.
        :param pathParametersList: [pathParameters] if 2-D or [outerPathParameters, innerPathParameters] if 3-D
        :param elementsCountAround: Number of elements around this segment.
        :param elementsCountThroughWall: Number of elements between inner and outer tube if 3-D, 1 if 2-D.
        :param isCore: True for generating a solid core inside the tube, False for regular tube network.
        :param elementsCountAcrossMajor: Number of elements across major axis of an ellipse.
        :param elementsCountAcrossMinor: Number of elements across minor axis of an ellipse.
        :param elementsCountTranstion: Number of elements across transition zone between core box elements and
        rim elements.
        """
        super(TubeNetworkMeshSegment, self).__init__(networkSegment, pathParametersList)
        self._isCore = isCore
        self._elementsCountAround = elementsCountAround
        self._elementsCountAcrossMajor = elementsCountAcrossMajor
        self._elementsCountAcrossMinor = elementsCountAcrossMinor
        self._elementsCountTransition = elementsCountTransition
        if self._isCore and self._elementsCountTransition > 1:
            self._elementsCountAround = (elementsCountAround - 8 * (self._elementsCountTransition - 1))
        assert elementsCountThroughWall > 0
        self._elementsCountThroughWall = elementsCountThroughWall
        self._rawTubeCoordinatesList = []
        self._rawTrackSurfaceList = []
        for pathParameters in pathParametersList:
            px, pd1, pd2, pd12 = getPathRawTubeCoordinates(pathParameters, self._elementsCountAround)
            self._rawTubeCoordinatesList.append((px, pd1, pd2, pd12))
            nx, nd1, nd2, nd12 = [], [], [], []
            for i in range(len(px)):
                nx += px[i]
                nd1 += pd1[i]
                nd2 += pd2[i]
                nd12 += pd12[i]
            self._rawTrackSurfaceList.append(TrackSurface(len(px[0]), len(px) - 1, nx, nd1, nd2, nd12, loop1=True))
        # list[pathsCount][4] of sx, sd1, sd2, sd12; all [nAlong][nAround]:
        self._sampledTubeCoordinates = [None for p in range(self._pathsCount)]
        self._rimCoordinates = None
        self._rimNodeIds = None
        self._rimElementIds = None  # [e2][e3][e1]

        self._boxCoordinates = None
        self._transitionCoordinates = None
        self._boxNodeIds = None # [nAlong][nAcrossMajor][nAcrossMinor]
        self._boxBoundaryNodeIds = None
        # boxNodeIds that form the boundary of the solid core, rearranged in circular format
        self._boxBoundaryNodeToBoxId = None
        # lookup table that translates box boundary node ids in a circular format to box node ids in
        # [nAlong][nAcrossMajor][nAcrossMinor] format.

    def getElementsCountAround(self):
        return self._elementsCountAround

    def getRawTubeCoordinates(self, pathIndex=0):
        return self._rawTubeCoordinatesList[pathIndex]

    def getIsCore(self):
        return self._isCore

    def getElementsCountAcrossMajor(self):
        return self._elementsCountAcrossMajor

    def getElementsCountAcrossMinor(self):
        return self._elementsCountAcrossMinor

    def getElementsCountAcrossTransition(self):
        return self._elementsCountTransition

    def getRawTrackSurface(self, pathIndex=0):
        return self._rawTrackSurfaceList[pathIndex]

    def sample(self, targetElementLength):
        trimSurfaces = [self._junctions[j].getTrimSurfaces(self) for j in range(2)]
        minimumElementsCountAlong = 2 if (self._isLoop or ((self._junctions[0].getSegmentsCount() > 2) and
                (self._junctions[1].getSegmentsCount() > 2))) else 1
        elementsCountAlong = None
        for p in range(self._pathsCount):
            # determine elementsCountAlong for first/outer tube then fix for inner tubes
            self._sampledTubeCoordinates[p] = resampleTubeCoordinates(
                self._rawTubeCoordinatesList[p], fixedElementsCountAlong=elementsCountAlong,
                targetElementLength=targetElementLength, minimumElementsCountAlong=minimumElementsCountAlong,
                startSurface=trimSurfaces[0][p], endSurface=trimSurfaces[1][p])
            if not elementsCountAlong:
                elementsCountAlong = len(self._sampledTubeCoordinates[0][0]) - 1

        if self._dimension == 2:
            # copy first sampled tube coordinates, but insert single-entry 'n3' index after n2
            self._rimCoordinates = (
                [[ring] for ring in self._sampledTubeCoordinates[0][0]],
                [[ring] for ring in self._sampledTubeCoordinates[0][1]],
                [[ring] for ring in self._sampledTubeCoordinates[0][2]],
                None)
        else:
            wallFactor = 1.0 / self._elementsCountThroughWall
            ox, od1, od2 = self._sampledTubeCoordinates[0][0:3]
            ix, id1, id2 = self._sampledTubeCoordinates[1][0:3]
            rx, rd1, rd2, rd3 = [], [], [], []
            for n2 in range(elementsCountAlong + 1):
                coreCentre, arcCentre = self._determineCentrePoints(n2)
                for r in (rx, rd1, rd2, rd3):
                    r.append([])
                otx, otd1, otd2 = ox[n2], od1[n2], od2[n2]
                itx, itd1, itd2 = ix[n2], id1[n2], id2[n2]
                # wx, wd3 = self._determineWallCoordinates(otx, otd1, otd2, itx, itd1, itd2, coreCentre, arcCentre)
                wd3 = [mult(sub(otx[n1], itx[n1]), wallFactor) for n1 in range(self._elementsCountAround)]
                for n3 in range(self._elementsCountThroughWall + 1):
                    oFactor = n3 / self._elementsCountThroughWall
                    iFactor = 1.0 - oFactor
                    for r in (rx, rd1, rd2, rd3):
                        r[n2].append([])
                    for n1 in range(self._elementsCountAround):
                        if n3 == 0:
                            x, d1, d2 = itx[n1], itd1[n1], itd2[n1]
                        elif n3 == self._elementsCountThroughWall:
                            x, d1, d2 = otx[n1], otd1[n1], otd2[n1]
                        else:
                            x = add(mult(itx[n1], iFactor), mult(otx[n1], oFactor))
                            # x = wx[n3][n1]
                            d1 = add(mult(itd1[n1], iFactor), mult(otd1[n1], oFactor))
                            d2 = add(mult(itd2[n1], iFactor), mult(otd2[n1], oFactor))
                        d3 = wd3[n1]
                        # d3 = wd3[n3][n1]
                        for r, value in zip((rx, rd1, rd2, rd3), (x, d1, d2, d3)):
                            r[n2][n3].append(value)
            self._rimCoordinates = rx, rd1, rd2, rd3
        self._rimNodeIds = [None] * (elementsCountAlong + 1)
        self._rimElementIds = [None] * elementsCountAlong

        if self._isCore:
            # sample coordinates for the solid core
            self._sampleCoreCoordinates(elementsCountAlong)

    def _sampleCoreCoordinates(self, elementsCountAlong):
        """
        Black box function for sampling coordinates for the solid core.
        :param elementsCountAlong: A number of elements along a segment.
        """
        boxx, boxd1, boxd3 = [], [], []
        transx, transd1, transd3 = [], [], []
        for n2 in range(elementsCountAlong + 1):
            coreCentre, arcCentre = self._determineCentrePoints(n2)
            cbx, cbd1, cbd3, ctx, ctd1, ctd3 = self._generateCoreCoordinates(n2, coreCentre)
            for lst, value in zip((boxx, boxd1, boxd3, transx, transd1, transd3),
                                  (cbx, cbd1, cbd3, ctx, ctd1, ctd3)):
                lst.append(value)
        boxd2, transd2 = self._determineCoreD2Derivatives(boxx, boxd1, boxd3, transx, transd1, transd3)
        self._boxCoordinates = boxx, boxd1, boxd2, boxd3
        self._transitionCoordinates = transx, transd1, transd2, transd3
        self._boxNodeIds = [None] * (elementsCountAlong + 1)

    def _determineCentrePoints(self, n2):
        """
        Calculates coordinates for the centre of the solid core based on outer and inner tube coordinates.
        :param n2: Index for elements along the tube.
        :return: Coordinates of the solid core.
        """
        ox = self._sampledTubeCoordinates[0][0][n2]
        ix = self._sampledTubeCoordinates[1][0][n2]
        cp = []

        for x in [ox, ix]:
            P0 = x[self._elementsCountAround // 4]
            P1 = x[self._elementsCountAround // 4 * 3]
            midpoint = mult(add(P0, P1), 0.5)
            cp.append(midpoint)

        coreCentres, arcCentres = [], []
        for i in range(self._elementsCountAround):
            tol = 1e-10  # tolerance to avoid float division zero
            OP = magnitude(sub(ox[i], cp[0]))
            IP = magnitude(sub(ix[i], cp[1]))
            distBetweenOuterAndInner = magnitude(sub(cp[1], cp[0]))
            distBetweenOuterAndInner = tol if distBetweenOuterAndInner == 0 else distBetweenOuterAndInner
            outerBase = (OP ** 2 - IP ** 2 - distBetweenOuterAndInner ** 2) / (2 * distBetweenOuterAndInner)
            circularArcRadius = math.sqrt(outerBase ** 2 + OP ** 2)
            distBetweenCoreAndInner = circularArcRadius - outerBase - distBetweenOuterAndInner

            directionVector = sub(cp[1], cp[0])
            if directionVector[0] == 0 and directionVector[1] == 0 and directionVector[2] == 0:
                directionVector = [tol, tol, tol]
            scaledDV = mult(normalize(directionVector), (distBetweenOuterAndInner + distBetweenCoreAndInner))
            c = add(scaledDV, cp[0])
            coreCentres.append(c)

            dvi = [-d for d in directionVector]
            mag = magnitude(dvi)
            tol = 1e-10
            scaleDVI = mult(normalize(dvi), outerBase) if mag > tol else [0, 0, 0]
            ac = add(scaleDVI, cp[0])
            arcCentres.append(ac)

        coreCentre = [sum(e) / len(e) for e in zip(*coreCentres)]
        arcCentre = [sum(e) / len(e) for e in zip(*arcCentres)]

        return coreCentre, arcCentre

    def _createMirrorCurve(self, n2, centre):
        """
        Generate coordinates and derivatives for the mirror curve.
        :param n2: Index for elements along the tube.
        :param centre: Coordinates of the solid core.
        :return: Coordinates and derivatives for the mirror curve.
        """
        ix = self._rimCoordinates[0][n2][0]
        id3 = self._rimCoordinates[3][n2][0]

        # Create mirror curves
        n2a = 0
        n2z = self._elementsCountAround // 2

        rscx, rscd1 = [], []
        for n in range(2):
            startIndex = [n2a, n2z][n]
            tmdxStart = ix[startIndex] if n == 0 else centre
            tmdxEnd = centre if n == 0 else ix[startIndex]
            tmdd3 = id3[startIndex]
            rcx = [tmdxStart, tmdxEnd]
            mag = -1 if n == 0 else 1
            rcd3 = [set_magnitude(tmdd3, mag), set_magnitude(tmdd3, mag)]
            elementsCountAcross = self._elementsCountAcrossMajor // 2

            tx, td1 = sampleCubicHermiteCurves(rcx, rcd3, elementsCountAcross, arcLengthDerivatives=True)[0:2]

            rscx += tx[n::]
            rscd1 += td1[n::]

        rscd1 = smoothCubicHermiteDerivativesLine(rscx, rscd1, fixStartDirection=True, fixEndDirection=True)

        # Determine d3 derivatives
        rscd3 = []
        for n in range(len(rscx)):
            d3 = normalize(
                [ix[self._elementsCountAround // 4][c] - ix[(self._elementsCountAround // 4) * 3][c] for c in range(3)])
            rscd3.append(d3)

        return rscx, rscd1, rscd3

    def _sampleCoreNodesAlongMinorAxis(self, n2, rscx, rscd1, rscd3):
        """
        Samples nodes along minor axis (z-axis) between inner tube coordinate, centre and the opposing inner tube
        coordinate by the number of elements across minor axis.
        :param n2: Index for elements along the tube.
        :param rscx, rscd1, rscd3: Lists of coordinates, d1 and d3 derivatives for mirror curve.
        :return: Coordinates and derivatives for the solid core and the transition nodes around the core.
        """
        ix = self._rimCoordinates[0][n2][0]
        id1 = self._rimCoordinates[1][n2][0]
        id3 = self._rimCoordinates[3][n2][0]

        # Create an empty list for the core with dimensions of (major - 1) x (minor - 1)
        m = self._elementsCountAcrossMajor - 1 - 2 * (self._elementsCountTransition - 1)
        n = self._elementsCountAcrossMinor - 1 - 2 * (self._elementsCountTransition - 1)
        cbx, cbd1, cd2, cbd3, ctx, ctd1, ctd2, ctd3 = [], [], [], [], [], [], [], []

        for i in range(m):
            for lst in (cbx, cbd1, cd2, cbd3):
                lst.append([[0, 0, 0] for _ in range(n)])

        for i in range(self._elementsCountTransition - 1):
            for lst in (ctx, ctd1, ctd2, ctd3):
                lst.append([[0, 0, 0] for _ in range(self._elementsCountAround)])

        # Create Regular Row Curves
        n2a = 0
        n2b = self._elementsCountTransition - 1
        n2d = n1d = self._elementsCountAcrossMinor // 2 - n2b
        n2m = self._elementsCountAcrossMajor - (4 + 2 * n2b) + n2d
        n2z = self._elementsCountAcrossMinor

        ixTopHalf = ix[0:(self._elementsCountAround // 2) + 1]
        id1TopHalf = id1[0:(self._elementsCountAround // 2) + 1]
        id3TopHalf = id3[0:(self._elementsCountAround // 2) + 1]
        ixBtmHalf = (ix[(self._elementsCountAround // 2)::])[::-1]
        id1BtmHalf = (id1[(self._elementsCountAround // 2)::])[::-1]
        id3BtmHalf = (id3[(self._elementsCountAround // 2)::])[::-1]

        tx, td1, td3 = [], [], []
        trx, trd1, trd3 = [], [], []
        for n in range(n2m + 1 - n2d):
            n2 = list(range(n2d, n2m + 1))[n]
            c = self._elementsCountTransition + 1
            c2 = list(range(c, c + (n2m + 1 - n2d)))[n]
            nd1 = [set_magnitude(id3BtmHalf[n2 - 1], -1.0), set_magnitude(rscd3[c2], 1.0),
                   set_magnitude(id3TopHalf[n2], 1.0)]
            txm, td3m, pe, pxi, psf = sampleCubicHermiteCurves([ixBtmHalf[n2 - 1], rscx[c2], ixTopHalf[n2]], nd1,
                                                               self._elementsCountAcrossMinor,
                                                               arcLengthDerivatives=True)
            td1m = interpolateSampleCubicHermite([[-id1BtmHalf[n2 - 1][c] for c in range(3)], rscd1[c2],
                                                  id1TopHalf[n2]], [[0.0, 0.0, 0.0]] * 3, pe, pxi, psf)[0]

            td3m = smoothCubicHermiteDerivativesLine(txm, td3m, fixStartDirection=True, fixEndDirection=True)

            [lst1.append(lst2[n2a + 1: n2z]) for lst1, lst2 in zip((tx, td1, td3), (txm, td1m, td3m))]

        if self._elementsCountTransition > 1:
            for i in range(len(tx)):
                [lst.append([]) for lst in (trx, trd1, trd3)]
                for j in range(self._elementsCountTransition - 1):
                    trx[-1].append([tx[i].pop(0), tx[i].pop(-1)])
                    trd1[-1].append([td1[i].pop(0), td1[i].pop(-1)])
                    trd3[-1].append([td3[i].pop(0), td3[i].pop(-1)])

        # Store coordinate and derivative values into appropriate lists
        for n2c in range(1, self._elementsCountAcrossMajor - 2 - 2 * (self._elementsCountTransition - 1)):
            for n1 in range(len(tx[n2c - 1])):
                for lst, value in zip([cbx[n2c], cbd1[n2c], cbd3[n2c]],
                                      [tx[n2c - 1][n1], td1[n2c - 1][n1], td3[n2c - 1][n1]]):
                    lst[n1] = value

        if self._elementsCountTransition > 1:
            for n2 in range(self._elementsCountTransition - 1):
                for n in range(len(trx)):
                    n1 = -(n1d + n)
                    for lst, value in zip([ctx[n2], ctd1[n2], ctd3[n2]],
                                          [trx[n][n2][0], trd1[n][n2][0], trd3[n][n2][0]]):
                        lst[n1] = value
                for n in range(len(trx)):
                    n1 = n1d + n
                    for lst, value in zip([ctx[n2], ctd1[n2], ctd3[n2]],
                                          [trx[n][n2][-1], trd1[n][n2][-1], trd3[n][n2][-1]]):
                        lst[n1] = value

        return cbx, cbd1, cbd3, ctx, ctd1, ctd3

    def _sampleCoreNodeAlongMajorAxis(self, n2, cbx, cbd1, cbd3, ctx, ctd1, ctd3):
        """
        Samples nodes along major axis (y-axis) between inner tube coordinate, centre and the opposing inner tube
        coordinate by the number of elements across major axis.
        :param n2: Index for elements along the tube.
        :param cbx, cbd1, cbd3: Lists of coordinates, d1 and d3 derivatives for solid core.
        :param ctx, ctd1, ctd3: Lists of coordinates, d1 and d3 derivatives for rims around the core.
        :return: Coordinates and derivatives for the solid core and the rim(s) around the core.
        """
        ix, id1, id3 = self._rimCoordinates[0][n2][0], self._rimCoordinates[1][n2][0], self._rimCoordinates[3][n2][0]

        # Create regular column curves
        n1a, n1m, n1z = 0, self._elementsCountAround // 2, self._elementsCountAcrossMajor
        ec = (self._elementsCountAcrossMinor - 4) // 2 - (self._elementsCountTransition - 1)
        startIndexes = list(range(n1a - ec, n1a + ec + 1))
        endIndexes = list(range(n1m - ec, n1m + ec + 1))[::-1]
        tx, td1, td3, trx, trd1, trd3 = [], [], [], [], [], []

        elementsCountAcross = self._elementsCountAcrossMajor // 2
        nloop = (self._elementsCountAcrossMinor - 3 - 2 * (self._elementsCountTransition - 1)) \
            if self._elementsCountAcrossMinor > 4 else self._elementsCountAcrossMinor - 3  # This is affected by ellipse parameters issue

        for n in range(nloop):
            n1, n2 = startIndexes[n], endIndexes[n]
            m1 = n + 1
            m2 = (self._elementsCountAcrossMajor - 1 - 2 * (self._elementsCountTransition - 1)) // 2
            txm, td1m, td3m = [], [], []
            for n in range(2):
                if n == 0:
                    startx, startd1 = ix[n1], id1[n1]
                    endx, endd1, endd3 = cbx[m2][m1], cbd1[m2][m1], cbd3[m2][m1]
                    startd3 = [-id3[n1][c] for c in range(3)]
                    nd1 = [set_magnitude(startd3, 1), set_magnitude(endd1, 1)]
                    nd3 = [startd1, endd3]
                else:
                    startx, startd1, startd3 = cbx[m2][m1], cbd1[m2][m1], cbd3[m2][m1]
                    endx, endd3 = ix[n2], id3[n2]
                    endd1 = [-id1[n2][c] for c in range(3)]
                    nd1 = [set_magnitude(startd1, 1), set_magnitude(endd3, 1)]
                    nd3 = [startd3, endd1]

                tempx, tempd1, pe, pxi, psf = sampleCubicHermiteCurves([startx, endx], nd1,
                                                                       elementsCountAcross, arcLengthDerivatives=True)

                tempd3 = interpolateSampleCubicHermite(nd3, [[0.0, 0.0, 0.0]] * 2, pe, pxi, psf)[0]

                for lst, value in zip([txm, td1m, td3m], [tempx, tempd1, tempd3]):
                    lst += value[n::]

            td1m = smoothCubicHermiteDerivativesLine(txm, td1m, fixStartDirection=True, fixEndDirection=True)

            for lst, value in zip([tx, td1, td3], [txm, td1m, td3m]):
                lst.append(value[n1a + 1: n1z])

        if self._elementsCountTransition > 1:
            for i in range(len(tx)):
                [lst.append([]) for lst in (trx, trd1, trd3)]
                for j in range(self._elementsCountTransition - 1):
                    for lst, value in zip([trx[-1], trd1[-1], trd3[-1]], [tx[i], td1[i], td3[i]]):
                        lst.append([value.pop(0), value.pop(-1)])

        for n2c in range(len(tx[0])):
            for n1 in range(1, self._elementsCountAcrossMinor - 2 - 2 * (self._elementsCountTransition - 1)):
                for lst, value in zip([cbx[n2c], cbd1[n2c], cbd3[n2c]],
                                      [tx[n1 - 1][n2c], td1[n1 - 1][n2c], td3[n1 - 1][n2c]]):
                    lst[n1] = value

        if self._elementsCountTransition > 1:
            for n2 in range(self._elementsCountTransition - 1):
                for n in range(len(trx)):
                    n1 = startIndexes[n]
                    for lst, value in zip([ctx[n2], ctd1[n2], ctd3[n2]],
                                          [trx[n][n2][0], trd1[n][n2][0], trd3[n][n2][0]]):
                        lst[n1] = value
                for n in range(len(trx)):
                    n1 = endIndexes[n]
                    for lst, value in zip([ctx[n2], ctd1[n2], ctd3[n2]],
                                          [trx[n][n2][-1], trd1[n][n2][-1], trd3[n][n2][-1]]):
                        lst[n1] = value

        # Fix derivatives for core rim
        nSkip = self._elementsCountAcrossMinor // 2 - (1 + self._elementsCountTransition)
        if ctx:
            for n2 in range(self._elementsCountTransition - 1):
                for n1 in range(-nSkip, nSkip + 1):
                    tempd1, tempd3 = copy.copy(ctd1[n2][n1]), copy.copy(ctd3[n2][n1])
                    tempd1 = [-tempd1[c] for c in range(3)]
                    ctd1[n2][n1], ctd3[n2][n1] = tempd3, tempd1
                for n1 in range(self._elementsCountAround // 2 - nSkip, self._elementsCountAround // 2 + nSkip + 1):
                    tempd1, tempd3 = copy.copy(ctd1[n2][n1]), copy.copy(ctd3[n2][n1])
                    tempd3 = [-tempd3[c] for c in range(3)]
                    ctd1[n2][n1], ctd3[n2][n1] = tempd3, tempd1
                for n1 in range(self._elementsCountAround // 4 * 3 - nSkip,
                                self._elementsCountAround // 4 * 3 + nSkip + 1):
                    tempd1, tempd3 = copy.copy(ctd1[n2][n1]), copy.copy(ctd3[n2][n1])
                    tempd1, tempd3 = [-tempd1[c] for c in range(3)], [-tempd3[c] for c in range(3)]
                    ctd3[n2][n1], ctd1[n2][n1] = tempd3, tempd1

        # Re-order the order of sublists, i.e. from inner to outer layer.
        ctx = ctx[::-1]

        return cbx, cbd1, cbd3, ctx, ctd1, ctd3

    def _determineCoreTriplePoints(self, n2, cbx, cbd1, cbd3, ctx, ctd1, ctd3):
        """
        Compute coordinates and derivatives of points where 3 square elements merge.
        :param n2: Index for elements along the tube.
        :param cbx, cbd1, cbd3: Coordinates, d1 and d3 derivatives of core box.
        :param ctx, ctd1, ctd3: Coordinates, d1 and d3 derivatives of core rim(s).
        :return: Coordinates and derivatives of box and rim components of the core.
        """
        ix, id1, id3 = self._rimCoordinates[0][n2][0], self._rimCoordinates[1][n2][0], self._rimCoordinates[3][n2][0]

        n1m = self._elementsCountAround // 2
        n2a = 0
        cix, cid1, cid3 = (ctx + [ix], ctd1 + [id1], ctd3 + [id3])
        minorSkip = self._elementsCountAcrossMinor // 2 - (1 + self._elementsCountTransition)
        majorSkip = self._elementsCountAcrossMajor // 2 - (1 + self._elementsCountTransition)

        # Generate triple points for the core box
        # Left top and bottom
        for n in range(2):
            ltx = []
            scalefactor = 1 if n == 0 else -1
            n1 = [-minorSkip, n1m + minorSkip][n]
            n1a = 1
            n1z = self._elementsCountAround // 4 * 3 + (majorSkip * scalefactor)
            n2 = [n2a, -1][n]
            n2b = [1, -2][n]
            tx, td1 = sampleCubicHermiteCurves([cix[0][n1], cbx[n2b][0]],
                                               [[(-cid1[0][n1][c] * scalefactor - cid3[0][n1][c]) for c in range(3)],
                                                set_magnitude(cbd1[n2b][0], scalefactor)], 2, arcLengthDerivatives=True)[
                      0:2]
            ltx.append(tx[1])
            tx, td1 = sampleCubicHermiteCurves([cix[0][n1z], cbx[n2][n1a]],
                                               [[(cid1[0][n1z][c] * scalefactor - cid3[0][n1z][c]) for c in range(3)],
                                                cbd3[n2][n1a]],
                                               2, arcLengthDerivatives=True)[0:2]
            ltx.append(tx[1])
            x = [(ltx[0][c] + ltx[1][c]) / 2.0 for c in range(3)]
            cbx[n2][0] = x
            cbd3[n2][0] = [(cbx[n2][1][c] - cbx[n2][0][c]) for c in range(3)]
            cbd1[n2][0] = [(cbx[n2 + scalefactor][0][c] - cbx[n2][0][c]) * scalefactor for c in range(3)]
        # Right
        for n in range(2):
            rtx = []
            scalefactor = 1 if n == 0 else -1
            n1 = [minorSkip, n1m - minorSkip][n]
            n1b = -2
            n2 = [n2a, -1][n]
            n2b = [1, -2][n]
            n1z = self._elementsCountAround // 4 - (majorSkip * scalefactor)
            tx, td1 = sampleCubicHermiteCurves([cix[0][n1], cbx[n2b][-1]],
                                               [[(cid1[0][n1][c] * scalefactor - cid3[0][n1][c]) for c in range(3)],
                                                [d * scalefactor for d in cbd1[n2b][-1]]], 2,
                                               arcLengthDerivatives=True)[0:2]
            rtx.append(tx[1])
            tx, td1 = sampleCubicHermiteCurves([cix[0][n1z], cbx[n2][n1b]],
                                               [[(-cid1[0][n1z][c] * scalefactor - cid3[0][n1z][c]) for c in range(3)],
                                                [-d for d in cbd3[n2][n1b]]], 2, arcLengthDerivatives=True)[0:2]
            rtx.append(tx[1])
            x = [(rtx[0][c] + rtx[1][c]) / 2.0 for c in range(3)]
            n2c = [0, - 1][n]
            cbx[n2][-1] = x
            cbd3[n2][-1] = [(cbx[n2c][-1][c] - cbx[n2c][-2][c]) for c in range(3)]
            cbd1[n2][-1] = [(cbx[n2b][-1][c] - cbx[n2c][-1][c]) * scalefactor for c in range(3)]

        # Sample nodes along triple points
        if self._elementsCountTransition > 1:
            # Left
            for n in range(2):
                scalefactor = 1 if n == 0 else -1
                n1a = [-minorSkip, n1m + minorSkip][n]
                n1c = [n1a - 1, n1a + 1][n]
                n2 = [0, -1][n]
                txm, td3m, pe, pxi, psf = sampleCubicHermiteCurves([ix[n1c], cbx[n2][0]],
                                                                   [[-id3[n1c][c] for c in range(3)],
                                                                    [cbd1[n2][0][c] * scalefactor + cbd3[n2][0][c] for c
                                                                     in range(3)]],
                                                                   self._elementsCountTransition,
                                                                   arcLengthDerivatives=True)
                td1m = interpolateSampleCubicHermite(
                    [id1[n1c], ([-cbd1[n2][0][c] + cbd3[n2][0][c] * scalefactor for c in range(3)])],
                    [[0.0, 0.0, 0.0]] * 2, pe, pxi, psf)[0]

                txm, td1m, td3m = txm[::-1], td1m[::-1], td3m[::-1]
                for n3 in range(self._elementsCountTransition - 1):
                    ctx[n3][n1c] = txm[n3 + 1]
                    ctd1[n3][n1c] = td1m[n3 + 1]
                    ctd3[n3][n1c] = [-d for d in td3m[n3 + 1]]
            # Right
            for n in range(2):
                scalefactor = 1 if n == 0 else -1
                n1a = [minorSkip, n1m - minorSkip][n]
                n1c = [n1a + 1, n1a - 1][n]
                n2 = [0, -1][n]
                txm, td3m, pe, pxi, psf = sampleCubicHermiteCurves([ix[n1c], cbx[n2][-1]],
                                                                   [[-id3[n1c][c] for c in range(3)],
                                                                    [cbd1[n2][-1][c] * scalefactor - cbd3[n2][-1][c] for
                                                                     c in range(3)]],
                                                                   self._elementsCountTransition,
                                                                   arcLengthDerivatives=True)
                td1m = interpolateSampleCubicHermite([id1[n1c],
                                                      ([cbd1[n2][-1][c] + cbd3[n2][-1][c] * scalefactor for c in
                                                        range(3)])],
                                                     [[0.0, 0.0, 0.0]] * 2, pe, pxi, psf)[0]

                txm, td1m, td3m = txm[::-1], td1m[::-1], td3m[::-1]
                for n3 in range(self._elementsCountTransition - 1):
                    ctx[n3][n1c] = txm[n3 + 1]
                    ctd1[n3][n1c] = td1m[n3 + 1]
                    ctd3[n3][n1c] = [-d for d in td3m[n3 + 1]]

        return cbx, cbd1, cbd3, ctx, ctd1, ctd3

    def _smoothCoreDerivatives(self, cbx, cbd1, cbd3, ctx, ctd1, ctd3):
        """
        Smooth d1 and d3 derivatives of the solid core.
        :param cbx, cbd1, cbd3: Coordinates, d1 and d3 derivatives of the core box.
        :param ctx, ctd1, ctd3: Coordinates, d1 and d3 derivatives of the core rim.
        :return: Smoothed d1 and d3 derivatives of the solid core.
        """
        # Smooth core box rows
        for n2c in [0, -1]:
            cbd3[n2c][1:-1] = smoothCubicHermiteDerivativesLine(cbx[n2c], cbd3[n2c])[1:-1]

        # Smooth core box columns
        for n1 in [0, -1]:
            tx = []
            td1 = []
            for n2 in range(len(cbx)):
                tx.append(cbx[n2][n1])
                td1.append(cbd1[n2][n1])
            td1 = smoothCubicHermiteDerivativesLine(tx, td1)
            for n2 in range(1, len(td1)):
                cbd1[n2][n1] = td1[n2]

        # Smooth core transition derivatives
        if self._elementsCountTransition > 1:
            for m in range(self._elementsCountTransition - 1):
                ctxTopHalf = ctx[m][0:(self._elementsCountAround // 2) + 1]
                ctd1TopHalf = ctd1[m][0:(self._elementsCountAround // 2) + 1]
                ctxBtmHalf = (ctx[m][(self._elementsCountAround // 2)::])
                ctd1BtmHalf = (ctd1[m][(self._elementsCountAround // 2)::])
                ctxHalf = [ctxTopHalf, ctxBtmHalf]
                ctd1Half = [ctd1TopHalf, ctd1BtmHalf]

                for n in range(2):
                    ctd1m = smoothCubicHermiteDerivativesLine(ctxHalf[n], ctd1Half[n],
                                                              fixStartDirection=True, fixEndDirection=True)
                    if n == 0:
                        ctd1[m][0:(self._elementsCountAround // 2) + 1] = ctd1m
                    else:
                        ctd1[m][(self._elementsCountAround // 2)::] = ctd1m

        return cbd1, cbd3, ctd1, ctd3

    def _generateCoreCoordinates(self, n2, coreCentre):
        """
        Creates coordinates and derivatives for the solid core (and the rim(s) around the core, if exists) based on
        outer and inner tube coordinates.
        :param n2: Index for elements along the tube.
        :param coreCentre: Coordinates at the centre point of solid core
        :return: Lists of coordinates and derivatives for the solid core and rims around the core.
        """
        # Determine mirror curves
        rscx, rscd1, rscd3 = self._createMirrorCurve(n2, coreCentre)
        # Create regular row curves
        cbx, cbd1, cbd3, ctx, ctd1, ctd3 = self._sampleCoreNodesAlongMinorAxis(n2, rscx, rscd1, rscd3)
        # Create regular column curves
        cbx, cbd1, cbd3, ctx, ctd1, ctd3 = self._sampleCoreNodeAlongMajorAxis(n2, cbx, cbd1, cbd3, ctx, ctd1, ctd3)
        # Get triple points
        cbx, cbd1, cbd3, ctx, ctd1, ctd3 = self._determineCoreTriplePoints(n2, cbx, cbd1, cbd3, ctx, ctd1, ctd3)
        # Smooth derivatives
        cbd1, cbd3, ctd1, ctd3 = self._smoothCoreDerivatives(cbx, cbd1, cbd3, ctx, ctd1, ctd3)

        return cbx, cbd1, cbd3, ctx, ctd1, ctd3

    def _determineCoreD2Derivatives(self, boxx, boxd1, boxd3, transx, transd1, transd3):
        """
        Compute d2 derivatives for the solid core.
        :param boxx, boxd1, boxd3: Coordinates and derivatives (d1 & d3) of the core box nodes.
        :param transx, transd1, transd3: Coordinates and derivatives (d1 & d3) of the core transition nodes.
        :return: D2 derivatives of box and rim components of the core.
        """
        elementsCountAlong = len(boxx)
        nodesCountAcrossMajor = len(boxx[0])
        nodesCountAcrossMinor = len(boxx[0][0])

        boxd2 = [[[None for _ in range(nodesCountAcrossMinor)] for _ in range(nodesCountAcrossMajor)]
                 for _ in range(elementsCountAlong)]
        transd2 = [[[None for _ in range(self._elementsCountAround)] for _ in range(self._elementsCountTransition - 1)]
                   for _ in range(elementsCountAlong)]

        for m in range(nodesCountAcrossMajor):
            for n in range(nodesCountAcrossMinor):
                tx, td2 = [], []
                for n2 in range(elementsCountAlong):
                    x = boxx[n2][m][n]
                    d2 = cross(boxd3[n2][m][n], boxd1[n2][m][n])
                    tx.append(x)
                    td2.append(d2)
                td2 = smoothCubicHermiteDerivativesLine(tx, td2, fixStartDirection=False, fixEndDirection=False)
                for n2 in range(elementsCountAlong):
                    boxd2[n2][m][n] = td2[n2]

        if self._elementsCountTransition > 1:
            for n3 in range(self._elementsCountTransition - 1):
                for n1 in range(self._elementsCountAround):
                    tx, td2 = [], []
                    for n2 in range(elementsCountAlong):
                        x = transx[n2][n3][n1]
                        d2 = cross(transd3[n2][n3][n1], transd1[n2][n3][n1])
                        tx.append(x)
                        td2.append(d2)
                    td2 = smoothCubicHermiteDerivativesLine(tx, td2, fixStartDirection=False, fixEndDirection=True)
                    for n2 in range(elementsCountAlong):
                        transd2[n2][n3][n1] = td2[n2]

        return boxd2, transd2

    def _determineWallCoordinates(self, ox, od1, od2, ix, id1, id2, coreCentre, arcCentre):
        """
        Calculates rim coordinates and d3 derivatives based on the centre point of the solid core.
        :param ox, od1, od2: Coordinates and (d1 and d2) derivatives for outermost rim.
        :param ix, id1, id2: Coordinates and (d1 and d2) derivatives for innermost rim.
        :param coreCentre: Centre point of the solid core.
        :param arcCetnre: Centre point of the arc that passes through the core centre, inner rim and outer rim.
        :return: Coordinates and d3 derivatives for rim nodes.
        """
        wx, wd3 = [], []

        # check if the cross-section of cylinder is regular shaped or irregular.
        dist1a = sub(ix[0], coreCentre)
        dist1b = sub(ix[self._elementsCountAround // 2], coreCentre)
        dist2a = sub(ix[self._elementsCountAround // 4], coreCentre)
        dist2b = sub(ix[self._elementsCountAround // 4 * 3], coreCentre)
        tol = 1e-3
        if abs(magnitude(dist1a) - magnitude(dist1b)) > tol or \
                abs(magnitude(dist2a) - magnitude(dist2b)) > tol:
            isRegular = False
        else:
            isRegular = True

        # Calculate d3 derivatives
        tx, td3 = [], []
        for n1 in range(self._elementsCountAround):
            if isRegular:
                tol = 1e-10
                dist = sub(arcCentre, coreCentre)
                if magnitude(dist) > tol:
                    if dist > [tol, tol, tol]:
                        oc = sub(ox[n1], arcCentre)
                        ic = sub(ix[n1], arcCentre)
                    else:
                        oc = add(mult(oc[n1], -1), arcCentre)
                        ic = add(mult(ic[n1], -1), arcCentre)
                    ot = cross(oc, od1[n1])
                    it = cross(ic, id1[n1])
                else:
                    ot, it = cross(od1[n1], od2[n1]), cross(id1[n1], id2[n1])
                scalefactor = magnitude(sub(ox[n1], ix[n1])) / self._elementsCountThroughWall
                od3 = mult(normalize(ot), scalefactor)
                id3 = mult(normalize(it), scalefactor)
            else:
                wallFactor = 1.0 / self._elementsCountThroughWall
                od3 = id3 = mult(sub(ox[n1], ix[n1]), wallFactor)

            txm, td3m, pe, pxi, psf = sampleCubicHermiteCurves(
                [ix[n1], ox[n1]], [id3, od3], self._elementsCountThroughWall, arcLengthDerivatives=True)

            td3m = smoothCubicHermiteDerivativesLine(txm, td3m, fixStartDirection=True, fixEndDirection=True)

            tx.append(txm)
            td3.append(td3m)

        for n3 in range(self._elementsCountThroughWall + 1):
            wx.append([])
            wd3.append([])
            for n1 in range(self._elementsCountAround):
                wx[n3].append(tx[n1][n3])
                wd3[n3].append(td3[n1][n3])

        return wx, wd3

    def _createBoxBoundaryNodeIdsList(self, startSkipCount=None, endSkipCount=None):
        """
        Creates a list (in a circular format similar to other rim node id lists) of core box node ids that are
        located at the boundary of the core.
        This list is used to easily stitch inner rim nodes with box nodes.
        :param startSkipCount: Row in from start that node ids are for.
        :param endSkipCount: Row in from end that node ids are for.
        :return: A list of box node ids stored in a circular format, and a lookup list that translates indexes used in
        boxBoundaryNodeIds list to indexes that can be used in boxCoordinates list.
        """
        boxBoundaryNodeIds = []
        boxBoundaryNodeToBoxId = []
        elementsCountAlong = len(self._rimCoordinates[0]) - 1

        boxElementsCountRow = (self._elementsCountAcrossMajor - 2 * self._elementsCountTransition) + 1
        boxElementsCountColumn = (self._elementsCountAcrossMinor - 2 * self._elementsCountTransition) + 1
        for n2 in range(elementsCountAlong + 1):
            if (n2 < startSkipCount) or (n2 > elementsCountAlong - endSkipCount) or self._boxNodeIds[n2] is None:
                boxBoundaryNodeIds.append(None)
                boxBoundaryNodeToBoxId.append(None)
                continue
            else:
                boxBoundaryNodeIds.append([])
                boxBoundaryNodeToBoxId.append([])
            for n3 in range(boxElementsCountRow):
                if n3 == 0 or n3 == boxElementsCountRow - 1:
                    ids = self._boxNodeIds[n2][n3] if n3 == 0 else self._boxNodeIds[n2][n3][::-1]
                    n1List = list(range(boxElementsCountColumn)) if n3 == 0 else (
                        list(range(boxElementsCountColumn - 1, -1, -1)))
                    boxBoundaryNodeIds[n2] += [ids[c] for c in range(boxElementsCountColumn)]
                    for n1 in n1List:
                        boxBoundaryNodeToBoxId[n2].append([n3, n1])
                else:
                    for n1 in [-1, 0]:
                        boxBoundaryNodeIds[n2].append(self._boxNodeIds[n2][n3][n1])
                        boxBoundaryNodeToBoxId[n2].append([n3, n1])

            start = self._elementsCountAcrossMajor - 4 - 2 * (self._elementsCountTransition - 1)
            idx = self._elementsCountAcrossMinor - 2 * (self._elementsCountTransition - 1)
            for n in range(int(start), -1, -1):
                boxBoundaryNodeIds[n2].append(boxBoundaryNodeIds[n2].pop(idx + 2 * n))
                boxBoundaryNodeToBoxId[n2].append(boxBoundaryNodeToBoxId[n2].pop(idx + 2 * n))

            nloop = self._elementsCountAcrossMinor // 2 - self._elementsCountTransition
            for _ in range(nloop):
                boxBoundaryNodeIds[n2].insert(len(boxBoundaryNodeIds[n2]), boxBoundaryNodeIds[n2].pop(0))
                boxBoundaryNodeToBoxId[n2].insert(len(boxBoundaryNodeToBoxId[n2]),
                                                      boxBoundaryNodeToBoxId[n2].pop(0))

        return boxBoundaryNodeIds, boxBoundaryNodeToBoxId

    @classmethod
    def blendSampledCoordinates(cls, segment1, nodeIndexAlong1, segment2, nodeIndexAlong2):
        nodesCountAround = segment1._elementsCountAround
        nodesCountRim = len(segment1._rimCoordinates[0][0])
        if ((nodesCountAround != segment2._elementsCountAround) or
                (nodesCountRim != len(segment2._rimCoordinates[0][0]))):
            return  # can't blend unless these match

        # blend rim coordinates
        for n3 in range(nodesCountRim):
            s1d2 = segment1._rimCoordinates[2][nodeIndexAlong1][n3]
            s2d2 = segment2._rimCoordinates[2][nodeIndexAlong2][n3]
            for n1 in range(nodesCountAround):
                # harmonic mean magnitude
                s1d2Mag = magnitude(s1d2[n1])
                s2d2Mag = magnitude(s2d2[n1])
                d2Mag = 2.0 / ((1.0 / s1d2Mag) + (1.0 / s2d2Mag))
                d2 = mult(s1d2[n1], d2Mag / s1d2Mag)
                s1d2[n1] = d2
                s2d2[n1] = d2

    def getSampledElementsCountAlong(self):
        return len(self._sampledTubeCoordinates[0][0]) - 1

    def getSampledTubeCoordinatesRing(self, pathIndex, nodeIndexAlong):
        """
        Get a ring of sampled coordinates at the supplied node index.
        :param pathIndex: 0 for outer/primary, 1 or -1 for inner/secondary.
        :param nodeIndexAlong: Node index from 0 to self._elementsCountAlong, or negative to count from end.
        :return: sx[nAround]
        """
        return self._sampledTubeCoordinates[pathIndex][0][nodeIndexAlong]

    def getElementsCountRim(self):
        return max(1, len(self._rimCoordinates[0][0]) - 1)

    def getNodesCountRim(self):
        return len(self._rimCoordinates[0][0])

    def getRimCoordinatesListAlong(self, n1, n2List, n3):
        """
        Get list of parameters for n2 indexes along segment at given n1, n3.
        :param n1: Node index around segment.
        :param n2List: List of node indexes along segment.
        :param n3: Node index from inner to outer rim.
        :return: [x[], d1[], d2[], d3[]]. d3[] may be None
        """
        paramsList = []
        for i in range(4):
            params = []
            for n2 in n2List:
                params.append(self._rimCoordinates[i][n2][n3][n1] if self._rimCoordinates[i] else None)
            paramsList.append(params)
        return paramsList

    def getBoxCoordinatesListAlong(self, n1, n2List, n3):
        """
        Get a list of parameters for solid core box for n2 indexes along segment at given n1, n3.
        :param n1: Node index around segment.
        :param n2List: List of node indexes along segment.
        :param n3: Node index from inner to outer rim.
        :return: [x[], d1[], d2[], d3[]].
        """
        paramsList = []
        for i in range(4):
            params = []
            for n2 in n2List:
                params.append(self._boxCoordinates[i][n2][n3][n1] if self._boxCoordinates[i] else None)
            paramsList.append(params)

        return paramsList

    def getTransitionCoordinatesListAlong(self, n1, n2List, n3):
        """
        Get a list of parameters for core transition nodes for n2 indexes along segment at given n1, n3.
        :param n1: Node index around segment.
        :param n2List: List of node indexes along segment.
        :param n3: Node index from inner to outer rim.
        :return: [x[], d1[], d2[], d3[]].
        """
        paramsList = []
        for i in range(4):
            params = []
            for n2 in n2List:
                params.append(self._transitionCoordinates[i][n2][n3][n1] if self._transitionCoordinates[i]
                              else None)
            paramsList.append(params)

        return paramsList

    def getCoreNodesCountAcrossMajor(self):
        return len(self._boxCoordinates[0][0])

    def getCoreNodesCountAcrossMinor(self):
        return len(self._boxCoordinates[0][0][0])

    def getElementsCountTransition(self):
        return self._elementsCountTransition

    def getBoxCoordinates(self, n1, n2, n3):
        return (self._boxCoordinates[0][n2][n3][n1], self._boxCoordinates[1][n2][n3][n1],
                self._boxCoordinates[2][n2][n3][n1], self._boxCoordinates[3][n2][n3][n1])

    def getBoxNodeIds(self, n1, n2, n3):
        """
        Get a box node ID for n2 index along segment at given n1, n3.
        :param n1: Node index across major axis (y-axis).
        :param n2: Node index along segment.
        :param n3: Node index across minor axis (z-axis).
        :return: Node identifier.
        """
        return self._boxNodeIds[n2][n3][n1]

    def getBoxNodeIdsSlice(self, n2):
        """
        Get slice of box node IDs at n2 index along segment.
        :param n2: Node index along segment, including negative indexes from end.
        :return: Node IDs arrays, or None if not set.
        """
        return self._boxNodeIds[n2]

    def getBoxBoundaryNodeIds(self, n1, n2):
        """
        Get a node ID around the core box for n2 index along segment at a given n1.
        :param n1: Node index around the core box.
        :param n2: Node index along segment.
        :return: Node identifier.
        """
        return self._boxBoundaryNodeIds[n2][n1]

    def getBoxBoundaryNodeToBoxId(self, n1, n2):
        """
        Translate box boundary node indexes to core box node indexes in the form of major- and minor axes.
        :param n1: Node index around the core box.
        :param n2: Node index along segment.
        :return: n3 (major axis) and n1 (minor axis) indexes used in boxCoordinates.
        """
        return self._boxBoundaryNodeToBoxId[n2][n1]

    def getTriplePointIndexes(self):
        """
        Get a node ID at triple points (special four corners) of the solid core.
        :return: A list of circular (n1) indexes used to identify triple points.
        """
        elementsCountAround = self._elementsCountAround
        nodesCountAcrossMinorHalf = self.getCoreNodesCountAcrossMinor() // 2
        triplePointIndexesList = []

        for n in range(0, elementsCountAround, elementsCountAround // 2):
            triplePointIndexesList.append(n + nodesCountAcrossMinorHalf)
            triplePointIndexesList.append((n - nodesCountAcrossMinorHalf) % elementsCountAround)

        return triplePointIndexesList

    def getTriplePointLocation(self, e1):
        """
        Determines the location of a specific triple point relative to the solid core box.
        There are four locations: Top left (location = 1); top right (location = -1); bottom left (location = 2);
        and bottom right (location = -2). Location is None if not located at any of the four specified locations.
        :return: Location identifier.
        """
        em = (self._elementsCountAcrossMinor - 2) // 2 - (self._elementsCountTransition - 1)
        eM = (self._elementsCountAcrossMajor - 2) // 2 - (self._elementsCountTransition - 1)
        ec = self._elementsCountAround // 4

        lftColumnElements = list(range(0, ec - eM)) + list(range(3 * ec + eM, self._elementsCountAround))
        topRowElements = list(range(ec - eM, ec + eM))
        rhtColumnElements = list((range(2 * ec - em, 2 * ec + em)))
        btmRowElements = list(range(3 * ec - eM, 3 * ec + eM))

        idx = len(lftColumnElements) // 2
        if e1 == topRowElements[0] or e1 == lftColumnElements[idx - 1]:
            location = 1  # "TopLeft"
        elif e1 == topRowElements[-1] or e1 == rhtColumnElements[0]:
            location = -1  # "TopRight"
        elif e1 == btmRowElements[-1] or e1 == lftColumnElements[idx]:
            location = 2  # "BottomLeft"
        elif e1 == btmRowElements[0] or e1 == rhtColumnElements[-1]:
            location = -2  # "BottomRight"
        else:
            location = 0

        return location

    def getRimCoordinates(self, n1, n2, n3):
        """
        Get rim parameters at a point.
        :param n1: Node index around.
        :param n2: Node index along segment.
        :param n3: Node index from inner to outer rim.
        :return: x, d1, d2, d3
        """
        return (self._rimCoordinates[0][n2][n3][n1],
                self._rimCoordinates[1][n2][n3][n1],
                self._rimCoordinates[2][n2][n3][n1],
                self._rimCoordinates[3][n2][n3][n1] if self._rimCoordinates[3] else None)

    def getRimNodeId(self, n1, n2, n3):
        """
        Get a rim node ID for a point.
        :param n1: Node index around.
        :param n2: Node index along segment.
        :param n3: Node index from inner to outer rim.
        :return: Node identifier.
        """
        return self._rimNodeIds[n2][n3][n1]

    def getRimElementId(self, e1, e2, e3):
        """
        Get a rim element ID.
        :param e1: Element index around.
        :param e2: Element index along segment.
        :param e3: Element index from inner to outer rim.
        :return: Element identifier.
        """
        return self._rimElementIds[e2][e3][e1]

    def setRimElementId(self, e1, e2, e3, elementIdentifier):
        """
        Set a rim element ID. Only called by adjacent junctions.
        :param e1: Element index around.
        :param e2: Element index along segment.
        :param e3: Element index from inner to outer rim.
        :param elementIdentifier: Element identifier.
        """
        if not self._rimElementIds[e2]:
            elementsCountRim = self.getElementsCountRim()
            self._rimElementIds[e2] = [[None] * self._elementsCountAround for _ in range(elementsCountRim)]
        self._rimElementIds[e2][e3][e1] = elementIdentifier

    def getRimNodeIdsSlice(self, n2):
        """
        Get slice of rim node IDs.
        :param n2: Node index along segment, including negative indexes from end.
        :return: Node IDs arrays through wall and around, or None if not set.
        """
        return self._rimNodeIds[n2]

    def generateMesh(self, generateData: TubeNetworkMeshGenerateData, n2Only=None):
        """
        :param n2Only: If set, create nodes only for that single n2 index along. Must be >= 0!
        """
        elementsCountAlong = len(self._rimCoordinates[0]) - 1
        elementsCountRim = self.getElementsCountRim()
        elementsCountTransition = self.getElementsCountTransition()
        coordinates = generateData.getCoordinates()
        fieldcache = generateData.getFieldcache()
        startSkipCount = 1 if (self._junctions[0].getSegmentsCount() > 2) else 0
        endSkipCount = 1 if (self._junctions[1].getSegmentsCount() > 2) else 0

        # create nodes
        nodes = generateData.getNodes()
        isLinearThroughWall = generateData.isLinearThroughWall()
        nodetemplate = generateData.getNodetemplate()
        for n2 in range(elementsCountAlong + 1) if (n2Only is None) else [n2Only]:
            if (n2 < startSkipCount) or (n2 > elementsCountAlong - endSkipCount):
                if self._isCore:
                    self._boxNodeIds[n2] = None
                self._rimNodeIds[n2] = None
                continue
            if self._isCore:
                if self._rimNodeIds[n2] and self._boxNodeIds[n2]:
                    continue
            else:
                if self._rimNodeIds[n2]:
                    continue
            # get shared nodes from single adjacent segment, including loop on itself
            # only handles one in, one out
            if n2 == 0:
                if self._junctions[0].getSegmentsCount() == 2:
                    segments = self._junctions[0].getSegments()
                    if self._isCore:
                        boxNodeIds = segments[0].getBoxNodeIdsSlice(-1)
                        if boxNodeIds:
                            self._boxNodeIds[n2] = boxNodeIds
                    rimNodeIds = segments[0].getRimNodeIdsSlice(-1)
                    if rimNodeIds:
                        self._rimNodeIds[n2] = rimNodeIds
                        continue
            if n2 == elementsCountAlong:
                if self._junctions[1].getSegmentsCount() == 2:
                    segments = self._junctions[1].getSegments()
                    if self._isCore:
                        boxNodeIds = segments[1].getBoxNodeIdsSlice(0)
                        if boxNodeIds:
                            self._boxNodeIds[n2] = boxNodeIds
                    rimNodeIds = segments[1].getRimNodeIdsSlice(0)
                    if rimNodeIds:
                        self._rimNodeIds[n2] = rimNodeIds
                        continue

            # create core box nodes
            if self._boxCoordinates:
                self._boxNodeIds[n2] = [] if self._boxNodeIds[n2] is None else self._boxNodeIds[n2]
                nodesCountAcrossMajor = self.getCoreNodesCountAcrossMajor()
                nodesCountAcrossMinor = self.getCoreNodesCountAcrossMinor()
                for n3 in range(nodesCountAcrossMajor):
                    self._boxNodeIds[n2].append([])
                    rx = self._boxCoordinates[0][n2][n3]
                    rd1 = self._boxCoordinates[1][n2][n3]
                    rd2 = self._boxCoordinates[2][n2][n3]
                    rd3 = self._boxCoordinates[3][n2][n3]
                    for n1 in range(nodesCountAcrossMinor):
                        nodeIdentifier = generateData.nextNodeIdentifier()
                        node = nodes.createNode(nodeIdentifier, nodetemplate)
                        fieldcache.setNode(node)
                        for nodeValue, rValue in zip([Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                                                      Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3],
                                                     [rx[n1], rd1[n1], rd2[n1], rd3[n1]]):
                            coordinates.setNodeParameters(fieldcache, -1, nodeValue, 1, rValue)
                        self._boxNodeIds[n2][n3].append(nodeIdentifier)

            # create rim nodes and transition nodes (if there are more than 1 layer of transition)
            nodesCountRim = len(self._rimCoordinates[0][0])
            self._rimNodeIds[n2] = [] if self._rimNodeIds[n2] is None else self._rimNodeIds[n2]
            nloop = nodesCountRim + (elementsCountTransition - 1) if self._isCore else nodesCountRim
            for n3 in range(nloop):
                n3p = n3 - (elementsCountTransition - 1) if self._isCore else n3
                if self._isCore and elementsCountTransition > 1 and n3 < (elementsCountTransition - 1):
                    # transition coordinates
                    rx = self._transitionCoordinates[0][n2][n3]
                    rd1 = self._transitionCoordinates[1][n2][n3]
                    rd2 = self._transitionCoordinates[2][n2][n3]
                    rd3 = self._transitionCoordinates[3][n2][n3]
                else:
                    # rim coordinates
                    rx = self._rimCoordinates[0][n2][n3p]
                    rd1 = self._rimCoordinates[1][n2][n3p]
                    rd2 = self._rimCoordinates[2][n2][n3p]
                    rd3 = None if isLinearThroughWall else self._rimCoordinates[3][n2][n3p]
                ringNodeIds = []
                for n1 in range(self._elementsCountAround):
                    nodeIdentifier = generateData.nextNodeIdentifier()
                    node = nodes.createNode(nodeIdentifier, nodetemplate)
                    fieldcache.setNode(node)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, rx[n1])
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, rd1[n1])
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, rd2[n1])
                    if rd3:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, rd3[n1])
                    ringNodeIds.append(nodeIdentifier)
                self._rimNodeIds[n2].append(ringNodeIds)

        # create a new list containing box node ids are located at the boundary
        if self._isCore:
            self._boxBoundaryNodeIds, self._boxBoundaryNodeToBoxId = (
                self._createBoxBoundaryNodeIdsList(startSkipCount, endSkipCount))

        if n2Only is not None:
            return

        # create elements
        annotationMeshGroups = generateData.getAnnotationMeshGroups(self._annotationTerms)
        mesh = generateData.getMesh()
        elementtemplateStd = generateData.getStandardElementtemplate()
        eftStd = generateData.getStandardEft()
        for e2 in range(startSkipCount, elementsCountAlong - endSkipCount):
            self._rimElementIds[e2] = []
            e2p = e2 + 1
            if self._isCore:
                # create box elements
                elementsCountAcrossMinor = self.getCoreNodesCountAcrossMinor() - 1
                elementsCountAcrossMajor = self.getCoreNodesCountAcrossMajor() - 1
                for e3 in range(elementsCountAcrossMajor):
                    e3p = e3 + 1
                    for e1 in range(elementsCountAcrossMinor):
                        nids = []
                        for n1 in [e1, e1 + 1]:
                            nids += [self._boxNodeIds[e2][e3][n1], self._boxNodeIds[e2][e3p][n1],
                                     self._boxNodeIds[e2p][e3][n1], self._boxNodeIds[e2p][e3p][n1]]
                        elementIdentifier = generateData.nextElementIdentifier()
                        element = mesh.createElement(elementIdentifier, elementtemplateStd)
                        element.setNodesByIdentifier(eftStd, nids)
                        for annotationMeshGroup in annotationMeshGroups:
                            annotationMeshGroup.addElement(element)

                # create core transition elements
                triplePointIndexesList = self.getTriplePointIndexes()
                eftList = [None] * self._elementsCountAround
                scalefactorsList = [None] * self._elementsCountAround
                ringElementIds = []
                for e1 in range(self._elementsCountAround):
                    nids, nodeParameters, nodeLayouts = [], [], []
                    n1p = (e1 + 1) % self._elementsCountAround
                    location = self.getTriplePointLocation(e1)
                    nodeLayoutTransition = generateData.getNodeLayoutTransition()
                    nodeLayoutTransitionTriplePoint = generateData.getNodeLayoutTransitionTriplePoint(location)
                    for n2 in [e2, e2 + 1]:
                        for n1 in [e1, n1p]:
                            nids += [self._boxBoundaryNodeIds[n2][n1]]
                            n3c, n1c = self._boxBoundaryNodeToBoxId[n2][n1]
                            nodeParameters.append(self.getBoxCoordinates(n1c, n2, n3c))
                            nodeLayouts.append(nodeLayoutTransitionTriplePoint if n1 in triplePointIndexesList else
                                               nodeLayoutTransition)
                    for n2 in [e2, e2 + 1]:
                        for n1 in [e1, n1p]:
                            nids += [self._rimNodeIds[n2][0][n1]]
                            nodeParameters.append(self.getRimCoordinates(n1, n2, 0))
                            nodeLayouts.append(None)
                    eft = eftList[e1]
                    scalefactors = scalefactorsList[e1]
                    if not eft:
                        eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                        eftList[e1] = eft
                        scalefactorsList[e1] = scalefactors
                    elementtemplate = mesh.createElementtemplate()
                    elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
                    elementtemplate.defineField(coordinates, -1, eft)
                    elementIdentifier = generateData.nextElementIdentifier()
                    element = mesh.createElement(elementIdentifier, elementtemplate)
                    element.setNodesByIdentifier(eft, nids)
                    if scalefactors:
                        element.setScaleFactors(eft, scalefactors)
                    for annotationMeshGroup in annotationMeshGroups:
                        annotationMeshGroup.addElement(element)
                    ringElementIds.append(elementIdentifier)
                self._rimElementIds[e2].append(ringElementIds)

            # create rim elements
            nloop = elementsCountRim + (self._elementsCountTransition - 1) if self._isCore else elementsCountRim
            for e3 in range(nloop):
                ringElementIds = []
                for e1 in range(self._elementsCountAround):
                    e1p = (e1 + 1) % self._elementsCountAround
                    nids = []
                    for n3 in [e3, e3 + 1] if (self._dimension == 3) else [0]:
                        nids += [self._rimNodeIds[e2][n3][e1], self._rimNodeIds[e2][n3][e1p],
                                 self._rimNodeIds[e2p][n3][e1], self._rimNodeIds[e2p][n3][e1p]]
                    elementIdentifier = generateData.nextElementIdentifier()
                    element = mesh.createElement(elementIdentifier, elementtemplateStd)
                    element.setNodesByIdentifier(eftStd, nids)
                    for annotationMeshGroup in annotationMeshGroups:
                        annotationMeshGroup.addElement(element)
                    ringElementIds.append(elementIdentifier)
                self._rimElementIds[e2].append(ringElementIds)


class TubeNetworkMeshJunction(NetworkMeshJunction):
    """
    Describes junction between multiple tube segments, some in, some out.
    """

    def __init__(self, inSegments: list, outSegments: list):
        """
        :param inSegments: List of inward TubeNetworkMeshSegment.
        :param outSegments: List of outward TubeNetworkMeshSegment.
        """
        super(TubeNetworkMeshJunction, self).__init__(inSegments, outSegments)
        pathsCount = self._segments[0].getPathsCount()
        self._trimSurfaces = [[None for p in range(pathsCount)] for s in range(self._segmentsCount)]
        self._calculateTrimSurfaces()
        # rim indexes are issued for interior points connected to 2 or more segment node indexes
        # based on the outer surface, and reused through the wall
        self._rimIndexToSegmentNodeList = []  # list[rim index] giving list[(segment number, node index around)]
        self._segmentNodeToRimIndex = []  # list[segment number][node index around] to rimIndex
        # rim coordinates sampled in the junction are indexed by n3 (through the wall) and 'rim index'
        self._rimCoordinates = None  # if set, (rx[], rd1[], rd2[], rd3[]) each over [n3][rim index]
        self._rimNodeIds = None  # if set, nodeIdentifier[n3][rim index]

        # parameters used for solid core
        self._isCore = self._segments[0].getIsCore()
        self._boxCoordinates = None # [nAlong][nAcrossMajor][nAcrossMinor]
        self._transitionCoordinates = None
        self._boxNodeIds = None
        self._biSequence = None # sequence for bifurcation - either [1, 2, 3] or [1, 3, 2]
        self._triSequence = None # sequence for trifurcation
        self._boxIndexToSegmentNodeList = []
        # list[box index] giving list[(segment number, node index across major axis, node index across minor axis)]
        self._segmentNodeToBoxIndex = []
        # list[segment number][node index across major axis][node index across minor axis] to boxIndex

    def _calculateTrimSurfaces(self):
        """
        Calculate surfaces for trimming adjacent segments so they can smoothly transition at junction.
        Algorithm gets 6 intersection points of longitudinal lines around each segment with other tubes or the end.
        These are joined to make a partial cone radiating back to the centre of the junction.
        Longitudinal lines start on the edge adjacent to the other segment most normal to it.
        """
        if self._segmentsCount < 3:
            return
        pathsCount = self._segments[0].getPathsCount()
        # get directions at end of segments' paths:
        outDirs = [[] for s in range(self._segmentsCount)]
        for s in range(self._segmentsCount):
            endIndex = -1 if self._segmentsIn[s] else 0
            for p in range(pathsCount):
                pathParameters = self._segments[s].getPathParameters(p)
                outDir = normalize(pathParameters[1][endIndex])
                if self._segmentsIn[s]:
                    outDir = [-d for d in outDir]
                outDirs[s].append(outDir)

        trimPointsCountAround = 6
        trimAngle = 2.0 * math.pi / trimPointsCountAround
        for s in range(self._segmentsCount):
            endIndex = -1 if self._segmentsIn[s] else 0
            for p in range(pathsCount):
                pathParameters = self._segments[s].getPathParameters(p)
                d2End = pathParameters[2][endIndex]
                d3End = pathParameters[4][endIndex]
                endEllipseNormal = normalize(cross(d2End, d3End))
                sOutDir = outDirs[s][p]
                # get phase angles and weights of other segments
                angles = []
                weights = []
                sumWeights = 0.0
                maxWeight = 0.0
                phaseAngle = None
                for os in range(self._segmentsCount):
                    if os == s:
                        continue
                    osOutDir = outDirs[os][p]
                    dx = dot(osOutDir, d2End)
                    dy = dot(osOutDir, d3End)
                    if (dx == 0.0) and (dy == 0.0):
                        angle = 0.0
                        weight = 0.0
                    else:
                        angle = math.atan2(dy, dx)
                        if angle < 0.0:
                            angle += 2.0 * math.pi
                        weight = math.pi - math.acos(dot(sOutDir, osOutDir))
                        if weight > maxWeight:
                            maxWeight = weight
                            phaseAngle = angle
                        sumWeights += weight
                    weights.append(weight)
                    angles.append(angle)
                # get correction to phase angle
                weightedSumDeltaAngles = 0.0
                for os in range(len(angles)):
                    angle = angles[os] - phaseAngle
                    if angle < 0.0:
                        angle += 2.0 * math.pi
                    nearestAngle = math.floor(angle / trimAngle + 0.5) * trimAngle
                    deltaAngle = nearestAngle - angle
                    weightedSumDeltaAngles += weights[os] * deltaAngle
                phaseAngle -= weightedSumDeltaAngles / sumWeights
                lx, ld1, ld2, ld12 = getPathRawTubeCoordinates(
                    pathParameters, trimPointsCountAround, radius=1.0, phaseAngle=phaseAngle)
                pointsCountAlong = len(pathParameters[0])

                # get coordinates and directions of intersection points of longitudinal lines and other track surfaces
                rx = []
                rd1 = []
                trim = False
                lowestMaxProportionFromEnd = 1.0
                for n1 in range(trimPointsCountAround):
                    cx = [lx[n2][n1] for n2 in range(pointsCountAlong)]
                    cd1 = [ld1[n2][n1] for n2 in range(pointsCountAlong)]
                    cd2 = [ld2[n2][n1] for n2 in range(pointsCountAlong)]
                    cd12 = [ld12[n2][n1] for n2 in range(pointsCountAlong)]
                    x = lx[endIndex][n1]
                    d1 = ld1[endIndex][n1]
                    maxProportionFromEnd = 0.0
                    # find intersection point with other segments which is furthest from end
                    for os in range(self._segmentsCount):
                        if os == s:
                            continue
                        otherSegment = self._segments[os]
                        otherTrackSurface = otherSegment.getRawTrackSurface(p)
                        otherSurfacePosition, curveLocation, isIntersection = \
                            otherTrackSurface.findNearestPositionOnCurve(
                                cx, cd2, loop=False, sampleEnds=False, sampleHalf=2 if self._segmentsIn[s] else 1)
                        if isIntersection:
                            proportion2 = (curveLocation[0] + curveLocation[1]) / (pointsCountAlong - 1)
                            proportionFromEnd = math.fabs(proportion2 - (1.0 if self._segmentsIn[s] else 0.0))
                            if proportionFromEnd > maxProportionFromEnd:
                                trim = True
                                x, d2 = evaluateCoordinatesOnCurve(cx, cd2, curveLocation, loop=False, derivative=True)
                                d1 = evaluateCoordinatesOnCurve(cd1, cd12, curveLocation, loop=False)
                                n = cross(d1, d2)  # normal to this surface
                                ox, od1, od2 = otherTrackSurface.evaluateCoordinates(
                                    otherSurfacePosition, derivatives=True)
                                on = cross(od1, od2)  # normal to other surface
                                d1 = cross(n, on)
                                maxProportionFromEnd = proportionFromEnd
                    if maxProportionFromEnd < lowestMaxProportionFromEnd:
                        lowestMaxProportionFromEnd = maxProportionFromEnd
                    rx.append(x)
                    rd1.append(d1)

                if trim:
                    # centre of trim surfaces is at lowestMaxProportionFromEnd
                    if lowestMaxProportionFromEnd == 0.0:
                        xCentre = pathParameters[0][endIndex]
                    else:
                        proportion = (1.0 - lowestMaxProportionFromEnd) if self._segmentsIn[s] \
                            else lowestMaxProportionFromEnd
                        e = int(proportion)
                        curveLocation = (e, proportion - e)
                        xCentre = evaluateCoordinatesOnCurve(pathParameters[0], pathParameters[1], curveLocation)
                    # ensure d1 directions go around in same direction as loop
                    for n1 in range(trimPointsCountAround):
                        d1 = rd1[n1]
                        if dot(endEllipseNormal, cross(sub(rx[n1], xCentre), d1)) < 0.0:
                            for c in range(3):
                                d1[c] = -d1[c]
                    rd1 = smoothCubicHermiteDerivativesLoop(rx, rd1, fixAllDirections=True,
                                                            magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
                    rd2 = [sub(rx[n1], xCentre) for n1 in range(trimPointsCountAround)]
                    rd12 = smoothCurveSideCrossDerivatives(rx, rd1, [rd2], loop=True)[0]
                    nx = []
                    nd1 = []
                    nd2 = []
                    nd12 = []
                    for factor in (0.75, 1.25):
                        for n1 in range(trimPointsCountAround):
                            d2 = sub(rx[n1], xCentre)
                            x = add(xCentre, mult(d2, factor))
                            d1 = mult(rd1[n1], factor)
                            d12 = mult(rd12[n1], factor)
                            nx.append(x)
                            nd1.append(d1)
                            nd2.append(d2)
                            nd12.append(d12)
                    trimSurface = TrackSurface(trimPointsCountAround, 1, nx, nd1, nd2, nd12, loop1=True)
                    self._trimSurfaces[s][p] = trimSurface

    def getTrimSurfaces(self, segment):
        """
        :param segment: TubeNetworkMeshSegment which must join at junction.
        :return: List of trim surfaces for paths of segment at junction.
        """
        return self._trimSurfaces[self._segments.index(segment)]

    def _sampleMidPoint(self, segmentsParameterLists):
        """
        Get mid-point coordinates and derivatives within junction from 2 or more segments' parameters.
        :param segmentsParameterLists: List over segment indexes s of [x, d1, d2, d3], each with 2 last parameters.
        d3 will be None for 2-D of bicubic-linear.
        :return: Mid-point x, d1, d2, d3. Derivative magnitudes will need smoothing.`
        """
        segmentsIn = [dot(sub(params[0][1], params[0][0]), params[2][1]) > 0.0 for params in segmentsParameterLists]
        segmentsCount = len(segmentsIn)
        assert segmentsCount > 1
        d3Defined = None not in segmentsParameterLists[0][3]
        # for each segment get inward parameters halfway between last 2 parameters
        xi = 0.5
        hx = []
        hd1 = []
        hd2 = []
        hn = []  # normal hd1 x hd2
        hd3 = [] if d3Defined else None
        for s in range(segmentsCount):
            params = segmentsParameterLists[s]
            hd2m = [params[2][i] if segmentsIn[s] else [-d for d in params[2][i]] for i in range(2)]
            hx.append(interpolateCubicHermite(params[0][0], hd2m[0], params[0][1], hd2m[1], xi))
            hd1.append(mult(add(params[1][0], params[1][1]), 0.5 if segmentsIn[s] else -0.5))
            hd2.append(interpolateCubicHermiteDerivative(params[0][0], hd2m[0], params[0][1], hd2m[1], xi))
            hn.append(normalize(cross(hd1[-1], hd2[-1])))
            if d3Defined:
                hd3.append(mult(add(params[3][0], params[3][1]), 0.5))
        # get lists of mid-point parameters for all segment permutations
        mx = []
        md1 = []
        md2 = []
        md3 = [] if d3Defined else None
        xi = 0.5
        sideFactor = 1.0
        outFactor = 0.5  # only used if sideFactor is non-zero
        for s1 in range(segmentsCount - 1):
            # fxs1 = segmentsParameterLists[s1][0][0]
            # fd2s1 = segmentsParameterLists[s1][2][0]
            # if segmentsIn[s1]:
            #     fd2s1 = [-d for d in fd2s1]
            for s2 in range(s1 + 1, segmentsCount):
                hd2s1 = hd2[s1]
                hd2s2 = [-d for d in hd2[s2]]
                if sideFactor > 0.0:
                    # compromise with direct connection respecting surface tangents
                    sideDirection = normalize(sub(hx[s2], hx[s1]))
                    side1d1 = dot(normalize(hd1[s1]), sideDirection)
                    side1d2 = dot(normalize(hd2[s1]), sideDirection)
                    side1 = add(mult(hd1[s1], side1d1), mult(hd2[s1], side1d2))
                    side2d1 = dot(normalize(hd1[s2]), sideDirection)
                    side2d2 = dot(normalize(hd2[s2]), sideDirection)
                    side2 = add(mult(hd1[s2], side2d1), mult(hd2[s2], side2d2))
                    sideScaling = computeCubicHermiteDerivativeScaling(hx[s1], side1, hx[s2], side2)
                    hd2s1 = add(mult(hd2s1, outFactor), mult(side1, sideScaling * sideFactor))
                    hd2s2 = add(mult(hd2s2, outFactor), mult(side2, sideScaling * sideFactor))
                scaling = computeCubicHermiteDerivativeScaling(hx[s1], hd2s1, hx[s2], hd2s2)
                hd2s1 = mult(hd2s1, scaling)
                hd2s2 = mult(hd2s2, scaling)
                cx = interpolateCubicHermite(hx[s1], hd2s1, hx[s2], hd2s2, xi)
                cd2 = interpolateCubicHermiteDerivative(hx[s1], hd2s1, hx[s2], hd2s2, xi)
                mx.append(cx)
                md1.append(mult(add(hd1[s1], [-d for d in hd1[s2]]), 0.5))
                md2.append(cd2)
                # smooth smx, smd2 with 2nd row from end coordinates and derivatives
                # fxs2 = segmentsParameterLists[s2][0][0]
                # fd2s2 = segmentsParameterLists[s2][2][0]
                # if not segmentsIn[s1]:
                #     fd2s2 = [-d for d in fd2s1]
                # tmd2 = smoothCubicHermiteDerivativesLine(
                #     [fxs1, cx, fxs2], [fd2s1, cd2, fd2s2], fixStartDerivative=True, fixEndDerivative=True)
                # md2.append(tmd2[1])
                if d3Defined:
                    md3.append(mult(add(hd3[s1], hd3[s2]), 0.5))
        if segmentsCount == 2:
            if not segmentsIn[0]:
                md1[0] = [-d for d in md1[0]]
                md2[0] = [-d for d in md2[0]]
            return mx[0], md1[0], md2[0], md3[0] if d3Defined else None
        mCount = len(mx)
        cx = [sum(x[c] for x in mx) / mCount for c in range(3)]
        cd3 = [sum(d3[c] for d3 in md3) / mCount for c in range(3)] if d3Defined else None
        ns12 = [0.0, 0.0, 0.0]
        for m in range(len(md1)):
            cp12 = normalize(cross(md1[m], md2[m]))
            for c in range(3):
                ns12[c] += cp12[c]
        ns12 = normalize(ns12)

        # get best fit directions with these 360/segmentsCount degrees apart

        # get preferred derivative at centre out to each segment
        rd = [interpolateLagrangeHermiteDerivative(
            cx, segmentsParameterLists[s][0][0], [-d for d in segmentsParameterLists[s][2][0]] if segmentsIn[s]
            else segmentsParameterLists[s][2][0], 0.0) for s in range(segmentsCount)]
        # get orthonormal axes with ns12, axis1 in direction of first preferred derivative
        axis1 = normalize(cross(cross(ns12, rd[0]), ns12))
        axis2 = cross(ns12, axis1)
        # get angles and sequence around normal, starting at axis1
        angles = [0.0]
        for s in range(1, segmentsCount):
            angle = math.atan2(dot(rd[s], axis2), dot(rd[s], axis1))
            angles.append((2.0 * math.pi + angle) if (angle < 0.0) else angle)
        angle = 0.0
        sequence = [0]
        for s in range(1, segmentsCount):
            nextAngle = math.pi * 4.0
            nexts = 0
            for t in range(1, segmentsCount):
                if angle < angles[t] < nextAngle:
                    nextAngle = angles[t]
                    nexts = t
            angle = nextAngle
            sequence.append(nexts)

        angleIncrement = 2.0 * math.pi / segmentsCount
        deltaAngle = 0.0
        angle = 0.0
        magSum = 0.0
        for s in range(1, segmentsCount):
            angle += angleIncrement
            deltaAngle += angles[sequence[s]] - angle
            magSum += 1.0 / magnitude(rd[sequence[s]])
        deltaAngle = deltaAngle / segmentsCount
        d2Mean = segmentsCount / magSum

        angle = deltaAngle
        sd = [None] * segmentsCount
        for s in range(segmentsCount):
            x1 = d2Mean * math.cos(angle)
            x2 = d2Mean * math.sin(angle)
            sd[sequence[s]] = add(mult(axis1, x1), mult(axis2, x2))
            angle += angleIncrement

        if segmentsCount == 3:
            # get through score for pairs of directions
            maxThroughScore = 0.0
            si = None
            for s1 in range(segmentsCount - 1):
                dir1 = normalize(segmentsParameterLists[s1][2][1])
                for s2 in range(s1 + 1, segmentsCount):
                    dir2 = normalize(segmentsParameterLists[s2][2][1])
                    throughScore = abs(dot(dir1, dir2))
                    if throughScore > maxThroughScore:
                        maxThroughScore = throughScore
                        si = (s1, s2)
            if maxThroughScore < 0.9:
                # maintain symmetry of bifurcations
                if (segmentsIn == [True, True, False]) or (segmentsIn == [False, False, True]):
                    si = (0, 1)
                elif (segmentsIn == [True, False, True]) or (segmentsIn == [False, True, False]):
                    si = (2, 0)
                else:
                    si = (1, 2)

        elif segmentsCount == 4:
            si = sequence[1:3]
        else:
            print("TubeNetworkMeshJunction._sampleMidPoint not fully implemented for segmentsCount =", segmentsCount)
            si = (1, 2)

        # get harmonic mean of d1 magnitude around midpoints
        magSum = 0.0
        for s in range(segmentsCount):
            magSum += 1.0 / magnitude(md1[s])
        d1Mean = segmentsCount / magSum
        # overall harmonic mean of derivatives to err on the low side for the diagonal derivatives
        dMean = 2.0 / (1.0 / d1Mean + 1.0 / d2Mean)

        if segmentsCount == 4:
            # use the equal spaced directions
            td = [mult(normalize(sd[j]), dMean) for j in si]
        else:
            # td = [sd[j] for j in si]
            # compromise between preferred directions rd and equal spaced directions sd
            td = [mult(normalize(add(rd[j], sd[j])), dMean) for j in si]
            if segmentsIn.count(True) == 2:
                # reverse so matches inward directions
                td = [[-c for c in d] for d in td]
            # td = [sub(d, mult(ns12, dot(d, ns12))) for d in td]

        cd1, cd2 = td
        if dot(cross(cd1, cd2), ns12) < 0.0:
            cd1, cd2 = cd2, cd1
        return cx, cd1, cd2, cd3

    def _determineJunctionSequence(self):
        """
        Determines the sequence of a junction. Currently only works for bifurcation and trifurcation.
        There are two possible sequences for bifurcation - [0,1,2] and [0,2,1]. Sequence [0,1,2] indicates the second
        segment is located on the right-hand side of the first segment, and [0,2,1] indicates the second segment is
        located on the left-hand side of the first segment.
        2    1             1    2
         \  /               \  /
          0     [0,1,2]      0    [0,2,1]
        For trifurcation, only + shape is currently supported, not tetrahedral. The function determines plus sequence
        [0,1,2,3], [0,1,3,2] or [0,2,3,1].
        """

        assert self._segmentsCount == 3 or self._segmentsCount == 4

        if self._segmentsCount == 3 and self._isCore:
            # Find sequence order for bifurcation - only applies when core option is enabled:
            # counter-clockwise sequence is [1, 2, 3] and clockwise sequence is [1, 3, 2]
            pCoord = []
            for s in range(self._segmentsCount):
                segment = self._segments[s]
                n3 = (-1 if self._segmentsIn[s] else 0) if s >= 1 else 0
                p = segment.getBoxCoordinates(0, -2 if self._segmentsIn[s] else 1, n3)[0]
                pCoord.append(p)
            dist12 = magnitude(sub(pCoord[1], pCoord[0]))
            dist13 = magnitude(sub(pCoord[2], pCoord[0]))
            self._biSequence = [0, 1, 2] if dist12 > dist13 else [0, 2, 1]
        elif self._segmentsCount == 4:
            # only support plus + shape for now, not tetrahedral
            # determine plus + sequence [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3] or [0, 2, 3, 1]
            outDirections = []
            for s in range(self._segmentsCount):
                d1 = self._segments[s].getPathParameters()[1][-1 if self._segmentsIn[s] else 0]
                outDirections.append(normalize([-d for d in d1] if self._segmentsIn[s] else d1))
            ns = None
            for s in range(1, self._segmentsCount):
                if dot(outDirections[0], outDirections[s]) > -0.9:
                    ns = cross(outDirections[0], outDirections[s])
                    break
            # get orthonormal axes with ns12, axis1 in direction of first preferred derivative
            axis1 = normalize(cross(cross(ns, outDirections[0]), ns))
            axis2 = normalize(cross(ns, axis1))
            # get angles and sequence around normal, starting at axis1
            angles = [0.0]
            for s in range(1, self._segmentsCount):
                angle = math.atan2(dot(outDirections[s], axis2), dot(outDirections[s], axis1))
                angles.append((2.0 * math.pi + angle) if (angle < 0.0) else angle)
            angle = 0.0
            sequence = [0]
            for s in range(1, self._segmentsCount):
                nextAngle = math.pi * 4.0
                nexts = 0
                for t in range(1, self._segmentsCount):
                    if angle < angles[t] < nextAngle:
                        nextAngle = angles[t]
                        nexts = t
                angle = nextAngle
                sequence.append(nexts)
            self._triSequence = sequence
        else:
            return

    def _sampleBifurcation(self, aroundCounts, acrossMajorCounts):
        """
        Blackbox function for sampling bifurcation coordinates. The rim coordinates are first sampled, then the box
        coordinates are sampled, if Core option is enabled.
        :param aroundCounts: Number of elements around the tube.
        :param acrossMajorCounts: Number of elements across major axis of the solid core.
        :return rimIndexesCount, boxIndexesCount: Total number of rimIndexes and boxIndexes, respectively.
        """
        assert self._segmentsCount == 3

        # sample rim junction
        rimIndexesCount = 0
        # numbers of elements directly connecting pairs of segments
        # sequence = [0, 1, 2]
        connectionCounts = [(aroundCounts[s] + aroundCounts[s - 2] - aroundCounts[s - 1]) // 2 for s in range(3)]
        for s in range(3):
            if (connectionCounts[s] < 1) or (aroundCounts[s] != (connectionCounts[s - 1] + connectionCounts[s])):
                print("Can't make tube bifurcation between elements counts around", aroundCounts)
                return

        self._rimIndexToSegmentNodeList = []  # list[rim index] giving list[(segment index, node index around)]
        self._segmentNodeToRimIndex = [[None] * aroundCounts[s] for s in range(self._segmentsCount)]
        for s1 in range(3):
            s2 = (s1 + 1) % 3
            startNodeIndex1 = (aroundCounts[s1] - connectionCounts[s1]) // 2
            startNodeIndex2 = connectionCounts[s1] // -2
            for n in range(connectionCounts[s1] + 1):
                nodeIndex1 = startNodeIndex1 + (n if self._segmentsIn[s1] else (connectionCounts[s1] - n))
                if self._segmentNodeToRimIndex[s1][nodeIndex1] is None:
                    nodeIndex2 = startNodeIndex2 + ((connectionCounts[s1] - n) if self._segmentsIn[s2] else n)
                    if self._segmentNodeToRimIndex[s2][nodeIndex2] is None:
                        rimIndex = rimIndexesCount
                        # keep list in order from lowest s
                        self._rimIndexToSegmentNodeList.append(
                            [[s1, nodeIndex1], [s2, nodeIndex2]] if (s1 < s2) else
                            [[s2, nodeIndex2], [s1, nodeIndex1]])
                        self._segmentNodeToRimIndex[s2][nodeIndex2] = rimIndex
                        rimIndexesCount += 1
                    else:
                        rimIndex = self._segmentNodeToRimIndex[s2][nodeIndex2]
                        # keep list in order from lowest s
                        segmentNodeList = self._rimIndexToSegmentNodeList[rimIndex]
                        index = 0
                        for i in range(len(segmentNodeList)):
                            if s1 < segmentNodeList[i][0]:
                                break
                            index += 1
                        segmentNodeList.insert(index, [s1, nodeIndex1])
                    self._segmentNodeToRimIndex[s1][nodeIndex1] = rimIndex

        # sample box junction
        boxIndexesCount = 0
        if self._isCore:
            sequence = self._biSequence
            connectionCounts = [(acrossMajorCounts[s] + acrossMajorCounts[s - 2] - acrossMajorCounts[s - 1]) // 2 for s
                                in range(3)]
            midIndexes = [connectionCounts[s] - 1 for s in range(3)]
            for s in range(3):
                if acrossMajorCounts[s] != (connectionCounts[s - 1] + connectionCounts[s]):
                    print("Can't make core bifurcation between elements counts across major axis", acrossMajorCounts)
                    return

            nodesCountAcrossMinor = self._segments[0].getCoreNodesCountAcrossMinor()
            nodesCountAcrossMajorList = [self._segments[s].getCoreNodesCountAcrossMajor() for s in range(3)]
            self._segmentNodeToBoxIndex = \
                [[[None for _ in range(nodesCountAcrossMinor)] for _ in range(nodesCountAcrossMajorList[s])]
                 for s in range(self._segmentsCount)]
            for s1 in range(3):
                s2 = (s1 + 1) % 3 if sequence == [0, 1, 2] else (s1 - 1) % 3
                midIndex1 = midIndexes[s1]
                midIndex2 = midIndexes[s2]
                for m in range(connectionCounts[s1]):
                    for n in range(nodesCountAcrossMinor):
                        m1 = (m + midIndex1) if self._segmentsIn[s1] else (midIndex1 - m) % connectionCounts[s1]
                        m2 = (midIndex2 - m) % connectionCounts[s2] if self._segmentsIn[s2] else (m + midIndex2)
                        if m1 == midIndex1:
                            indexGroup = [[0, midIndexes[0], n], [1, midIndexes[1], n], [2, midIndexes[2], n]]
                        else:
                            indexGroup = [[s1, m1, n], [s2, m2, n]] if s1 < s2 else [[s2, m2, n], [s1, m1, n]]
                        if indexGroup not in self._boxIndexToSegmentNodeList:
                            self._boxIndexToSegmentNodeList.append(indexGroup)
                            boxIndexesCount += 1

            for boxIndex in range(boxIndexesCount):
                segmentNodeList = self._boxIndexToSegmentNodeList[boxIndex]
                for segmentNode in segmentNodeList:
                    s, m, n = segmentNode
                    self._segmentNodeToBoxIndex[s][m][n] = boxIndex

        return rimIndexesCount, boxIndexesCount

    def _sampleTrifurcation(self, aroundCounts, acrossMajorCounts):
        """
        Blackbox function for sampling trifurcation coordinates. The rim coordinates are first sampled, then the box
        coordinates are sampled, if Core option is enabled.
        :param aroundCounts: Number of elements around the tube.
        :param acrossMajorCounts: Number of elements across major axis of the solid core.
        :return rimIndexesCount, boxIndexesCount: Total number of rimIndexes and boxIndexes, respectively.
        """
        assert self._segmentsCount == 4

        # sample rim junction
        rimIndexesCount = 0
        sequence = self._triSequence
        pairCount02 = aroundCounts[sequence[0]] + aroundCounts[sequence[2]]
        pairCount13 = aroundCounts[sequence[1]] + aroundCounts[sequence[3]]
        throughCount02 = ((pairCount02 - pairCount13) // 2) if (pairCount02 > pairCount13) else 0
        throughCount13 = ((pairCount13 - pairCount02) // 2) if (pairCount13 > pairCount02) else 0
        throughCounts = [throughCount02, throughCount13, throughCount02, throughCount13]
        # numbers of elements directly connecting pairs of segments
        freeAroundCounts = [aroundCounts[sequence[s]] - throughCounts[s] for s in range(self._segmentsCount)]
        if freeAroundCounts[0] == freeAroundCounts[2]:
            count03 = freeAroundCounts[3] // 2
            count12 = freeAroundCounts[1] // 2
            connectionCounts = [count03, count12, count12, count03]
        elif freeAroundCounts[1] == freeAroundCounts[3]:
            count03 = freeAroundCounts[0] // 2
            count12 = freeAroundCounts[2] // 2
            connectionCounts = [count03, count12, count12, count03]
        else:
            connectionCounts = [((freeAroundCounts[s] + freeAroundCounts[(s + 1) % self._segmentsCount]
                                  - freeAroundCounts[s - 1] + (s % 2)) // 2) for s in range(self._segmentsCount)]

        for s in range(self._segmentsCount):
            if (aroundCounts[sequence[s]] != (connectionCounts[s - 1] + throughCounts[s] + connectionCounts[s])):
                print("Can't make tube junction between elements counts around", aroundCounts)
                return

        self._rimIndexToSegmentNodeList = []  # list[rim index] giving list[(segment index, node index around)]
        self._segmentNodeToRimIndex = [[None] * aroundCounts[s] for s in range(self._segmentsCount)]
        for os1 in range(self._segmentsCount):
            os2 = (os1 + 1) % self._segmentsCount
            s1 = sequence[os1]
            s2 = sequence[os2]
            s3 = sequence[(os1 + 2) % self._segmentsCount]
            halfThroughCount = throughCounts[os1] // 2
            os1ConnectionCount = connectionCounts[os1]
            os2ConnectionCount = connectionCounts[os2]
            if self._segmentsIn[s1]:
                startNodeIndex1 = (aroundCounts[s1] - os1ConnectionCount) // 2
            else:
                startNodeIndex1 = os1ConnectionCount // -2
            if self._segmentsIn[s2]:
                startNodeIndex2 = os1ConnectionCount // -2
            else:
                startNodeIndex2 = (aroundCounts[s2] - os1ConnectionCount) // 2
            if self._segmentsIn[s3]:
                startNodeIndex3h = os2ConnectionCount // -2
                startNodeIndex3l = startNodeIndex3h - (os2ConnectionCount - os1ConnectionCount)
            else:
                startNodeIndex3l = (aroundCounts[s3] - os2ConnectionCount) // 2
                startNodeIndex3h = startNodeIndex3l + (os2ConnectionCount - os1ConnectionCount)

            for n in range(-halfThroughCount, os1ConnectionCount + 1 + halfThroughCount):
                n1 = startNodeIndex1 + (n if self._segmentsIn[s1] else (os1ConnectionCount - n))
                segmentIndexes = [s1]
                nodeIndexes = [n1 % aroundCounts[s1]]
                if 0 <= n <= os1ConnectionCount:
                    n2 = startNodeIndex2 + ((os1ConnectionCount - n) if self._segmentsIn[s2] else n)
                    segmentIndexes.append(s2)
                    nodeIndexes.append(n2 % aroundCounts[s2])
                if halfThroughCount and ((n <= 0) or (n >= os1ConnectionCount)):
                    n3 = ((startNodeIndex3l if n <= 0 else startNodeIndex3h) +
                          ((os2ConnectionCount - n) if self._segmentsIn[s3] else n))
                    segmentIndexes.append(s3)
                    nodeIndexes.append(n3 % aroundCounts[s3])

                rimIndex = None
                for i in range(len(segmentIndexes)):
                    ri = self._segmentNodeToRimIndex[segmentIndexes[i]][nodeIndexes[i]]
                    if ri is not None:
                        rimIndex = ri
                        break
                if rimIndex is None:
                    # new rim index
                    rimIndex = rimIndexesCount
                    rimIndexesCount += 1
                    segmentNodeList = []
                    self._rimIndexToSegmentNodeList.append(segmentNodeList)
                else:
                    segmentNodeList = self._rimIndexToSegmentNodeList[rimIndex]
                # build maps: rim index <--> segment index, node index
                for i in range(len(segmentIndexes)):
                    segmentIndex = segmentIndexes[i]
                    nodeIndex = nodeIndexes[i]
                    if self._segmentNodeToRimIndex[segmentIndex][nodeIndex] is None:
                        # keep segment node list in order from lowest segment index
                        index = 0
                        for j in range(len(segmentNodeList)):
                            if segmentIndex < segmentNodeList[j][0]:
                                break
                            index += 1
                        segmentNodeList.insert(index, [segmentIndex, nodeIndex])
                        self._segmentNodeToRimIndex[segmentIndex][nodeIndex] = rimIndex

        # sample box junction
        boxIndexesCount = 0
        if self._isCore:
            pCoord = []
            for s in range(self._segmentsCount):
                segment = self._segments[s]
                n3 = (-1 if self._segmentsIn[s] else 0) if s >= 1 else 0
                p = segment.getBoxCoordinates(0, -2 if self._segmentsIn[s] else 1, n3)[0]
                pCoord.append(p)
            dist12 = magnitude(sub(pCoord[sequence[1]], pCoord[0]))
            dist13 = magnitude(sub(pCoord[sequence[-1]], pCoord[0]))
            if sequence == [0, 1, 3, 2]:
                sequence = [0, 2, 3, 1] if dist12 < dist13 else sequence
            elif sequence == [0, 1, 2, 3]:
                sequence = [0, 2, 1, 3] if dist12 < dist13 else sequence

            pairCount02 = acrossMajorCounts[sequence[0]] + acrossMajorCounts[sequence[2]]
            pairCount13 = acrossMajorCounts[sequence[1]] + acrossMajorCounts[sequence[3]]
            throughCount02 = ((pairCount02 - pairCount13) // 2) if (pairCount02 > pairCount13) else 0
            throughCount13 = ((pairCount13 - pairCount02) // 2) if (pairCount13 > pairCount02) else 0
            throughCounts = [throughCount02, throughCount13, throughCount02, throughCount13]
            freeAcrossCounts = [acrossMajorCounts[sequence[s]] - throughCounts[s] for s in range(self._segmentsCount)]
            if freeAcrossCounts[0] == freeAcrossCounts[2]:
                count03 = freeAcrossCounts[3] // 2
                count12 = freeAcrossCounts[1] // 2
                connectionCounts = [count03, count12, count12, count03]
            elif freeAcrossCounts[1] == freeAcrossCounts[3]:
                count03 = freeAcrossCounts[0] // 2
                count12 = freeAcrossCounts[2] // 2
                connectionCounts = [count03, count12, count12, count03]
            else:
                connectionCounts = [((freeAcrossCounts[s] + freeAcrossCounts[(s + 1) % self._segmentsCount]
                                      - freeAcrossCounts[s - 1] + (s % 2)) // 2) for s in range(self._segmentsCount)]
            for s in range(self._segmentsCount):
                if (acrossMajorCounts[sequence[s]] != (
                        connectionCounts[s - 1] + throughCounts[s] + connectionCounts[s])):
                    print("Can't make tube junction between elements counts around", acrossMajorCounts)
                    return

            nodesCountAcrossMinor = self._segments[0].getCoreNodesCountAcrossMinor()
            nodesCountAcrossMajorList = [self._segments[s].getCoreNodesCountAcrossMajor() for s in range(4)]
            midIndexes = [nodesCountAcrossMajor // 2 for nodesCountAcrossMajor in nodesCountAcrossMajorList]
            halfThroughCounts = [throughCounts[s] // 2 for s in range(4)]

            connectingIndexesList = [[] for s in range(4)]
            for s in range(4):
                s1 = (s - 1) % self._segmentsCount
                s2 = (s + 1) % self._segmentsCount
                shift = (midIndexes[sequence[s2]] - midIndexes[sequence[s1]]) // 2
                midIndex = midIndexes[sequence[s]]
                halfThroughCount = halfThroughCounts[s]
                connectingIndexesList[sequence[s]] = \
                    ([midIndex - halfThroughCount, midIndex + halfThroughCount] if halfThroughCount \
                         else [midIndex + shift])

            self._boxIndexToSegmentNodeList = []
            self._segmentNodeToBoxIndex = \
                [[[None for _ in range(nodesCountAcrossMinor)] for _ in range(nodesCountAcrossMajorList[s])]
                 for s in range(self._segmentsCount)]

            for os1 in range(self._segmentsCount):
                os2 = (os1 + 1) % self._segmentsCount
                s1 = sequence[os1]
                s2 = sequence[os2]
                s3 = sequence[(os1 + 2) % self._segmentsCount]
                s4 = sequence[(os1 + 3) % self._segmentsCount]
                aStartNodeIndex = midIndexes[s1]
                nodesCountAcrossMajor1 = nodesCountAcrossMajorList[s1]
                nodesCountAcrossMajor2 = nodesCountAcrossMajorList[s2]
                connectingIndexes1 = connectingIndexesList[s1]
                connectingIndexes2 = connectingIndexesList[s2]
                connectingIndexes3 = connectingIndexesList[s3]
                throughIndexes = list(range(connectingIndexes1[0] + 1, connectingIndexes1[1])) \
                    if throughCounts[os1] else [None]

                for m in range(acrossMajorCounts[s1] // 2):
                    m1 = (m + aStartNodeIndex) if self._segmentsIn[s1] else (aStartNodeIndex - m) % connectionCounts[s1]
                    for n in range(nodesCountAcrossMinor):
                        if m1 in throughIndexes:
                            m3 = midIndexes[s3]
                            indexGroup = [[s1, m1, n], [s3, m3, n]] if s1 < s3 else [[s3, m3, n], [s1, m1, n]]
                        elif m1 in connectingIndexes1:
                            if all(v == 0 for v in throughCounts):
                                indexGroup = [[s1, m1, n], [s2, m1, n], [s3, m1, n], [s4, m1, n]]
                            else:
                                if throughCounts[os1]:
                                    i = 1 if self._segmentsIn[s1] else 0
                                    m2 = connectingIndexes2[0]
                                    m3 = connectingIndexes3[i]
                                    indexGroup = [[s1, m1, n], [s2, m2, n], [s3, m3, n]]
                        else:
                            if self._segmentsIn[s1] == self._segmentsIn[s2]:
                                if throughCounts[os1]:
                                    m2 = nodesCountAcrossMajor2 - m1 - 1
                                else:
                                    m2 = nodesCountAcrossMajor2 - (m1 + 1) if not self._segmentsIn[s1] else (
                                            nodesCountAcrossMajor1 - (m1 + 1))
                            else:
                                m2 = m1 - throughCounts[s1] if self._segmentsIn[s1] else m1
                            indexGroup = [[s1, m1, n], [s2, m2, n]] if s1 < s2 else [[s2, m2, n], [s1, m1, n]]
                        indexGroup = sorted(indexGroup, key=lambda x: (x[0], x[0]), reverse=False)
                        if indexGroup not in self._boxIndexToSegmentNodeList:
                            self._boxIndexToSegmentNodeList.append(indexGroup)
                            boxIndexesCount += 1

            for boxIndex in range(boxIndexesCount):
                segmentNodeList = self._boxIndexToSegmentNodeList[boxIndex]
                for segmentNode in segmentNodeList:
                    s, m, n = segmentNode
                    self._segmentNodeToBoxIndex[s][m][n] = boxIndex

        return rimIndexesCount, boxIndexesCount

    def _optimiseRimIndexes(self, aroundCounts, rimIndexesCount):
        """
        Iterates through a number of permutations to find the most optimised lookup table for rim indexes.
        :param aroundCounts: Number of elements around the tube.
        :param rimIndexesCount: Total number of rim indexes.
        """
        # get node indexes giving the lowest sum of distances between adjoining points on outer sampled tubes
        permutationCount = 1
        for count in aroundCounts:
            permutationCount *= count
        minIndexes = None
        minSum = None
        indexes = [0] * self._segmentsCount
        rings = [self._segments[s].getSampledTubeCoordinatesRing(0, -1 if self._segmentsIn[s] else 0)
                 for s in range(self._segmentsCount)]
        for p in range(permutationCount):
            sum = 0.0
            for rimIndex in range(rimIndexesCount):
                segmentNodeList = self._rimIndexToSegmentNodeList[rimIndex]
                sCount = len(segmentNodeList)
                for i in range(sCount - 1):
                    s1, n1 = segmentNodeList[i]
                    nodeIndex1 = (n1 + indexes[s1]) % aroundCounts[s1]
                    x1 = rings[s1][nodeIndex1]
                    for j in range(i + 1, sCount):
                        s2, n2 = segmentNodeList[j]
                        nodeIndex2 = (n2 + indexes[s2]) % aroundCounts[s2]
                        x2 = rings[s2][nodeIndex2]
                        sum += magnitude([x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]])
            if (minSum is None) or (sum < minSum):
                minIndexes = copy.copy(indexes)
                minSum = sum
            # permute through indexes:
            for s in range(self._segmentsCount):
                indexes[s] += 1
                if indexes[s] < aroundCounts[s]:
                    break
                indexes[s] = 0

        # offset node indexes by minIndexes
        for rimIndex in range(rimIndexesCount):
            segmentNodeList = self._rimIndexToSegmentNodeList[rimIndex]
            for segmentNode in segmentNodeList:
                s, n = segmentNode
                nodeIndex = (n + minIndexes[s]) % aroundCounts[s]
                self._segmentNodeToRimIndex[s][nodeIndex] = rimIndex
                segmentNode[1] = nodeIndex

    def sample(self, targetElementLength):
        """
        Blend sampled d2 derivatives across 2-segment junctions with the same version.
        Sample junction coordinates between second-from-end segment coordinates.
        :param targetElementLength: Ignored here as always 2 elements across junction.
        """
        if self._segmentsCount == 1:
            return
        if self._segmentsCount == 2:
            TubeNetworkMeshSegment.blendSampledCoordinates(
                self._segments[0], -1 if self._segmentsIn[0] else 0,
                self._segments[1], -1 if self._segmentsIn[1] else 0)
            return

        aroundCounts = [segment.getElementsCountAround() for segment in self._segments]
        acrossMajorCounts = [segment.getElementsCountAcrossMajor() for segment in self._segments]

        # determine junction sequence
        self._determineJunctionSequence()

        if self._segmentsCount == 3:
            rimIndexesCount, boxIndexesCount = self._sampleBifurcation(aroundCounts, acrossMajorCounts)

        elif self._segmentsCount == 4:
            rimIndexesCount, boxIndexesCount = self._sampleTrifurcation(aroundCounts, acrossMajorCounts)

        else:
            print("Tube network mesh not implemented for", self._segmentsCount, "segments at junction")
            return

        if not rimIndexesCount:
            return

        # optimise rim indexes
        self._optimiseRimIndexes(aroundCounts, rimIndexesCount)

        # sample rim coordinates
        elementsCountTransition = self._segments[0].getElementsCountTransition()
        nodesCountRim = self._segments[0].getNodesCountRim() + (elementsCountTransition - 1) if self._isCore else (
            self._segments[0].getNodesCountRim())
        rx, rd1, rd2, rd3 = [
            [[None] * rimIndexesCount for _ in range(nodesCountRim)] for i in range(4)]
        self._rimCoordinates = (rx, rd1, rd2, rd3)
        for n3 in range(nodesCountRim):
            n3p = n3 - (elementsCountTransition - 1) if self._isCore else n3
            for rimIndex in range(rimIndexesCount):
                segmentNodeList = self._rimIndexToSegmentNodeList[rimIndex]
                # segments have been ordered from lowest to highest s index
                segmentsParameterLists = []
                for s, n1 in segmentNodeList:
                    if self._isCore and n3 < (elementsCountTransition - 1):
                        segmentsParameterLists.append(
                            self._segments[s].getTransitionCoordinatesListAlong(
                                n1, [-2, -1] if self._segmentsIn[s] else [1, 0], n3))
                    else:
                        segmentsParameterLists.append(
                            self._segments[s].getRimCoordinatesListAlong(
                                n1, [-2, -1] if self._segmentsIn[s] else [1, 0], n3p))
                rx[n3][rimIndex], rd1[n3][rimIndex], rd2[n3][rimIndex], rd3[n3][rimIndex] = \
                    self._sampleMidPoint(segmentsParameterLists)

        # sample box coordinates
        if self._isCore:
            bx, bd1, bd2, bd3 = [[None] * boxIndexesCount for _ in range(4)]
            self._boxCoordinates = (bx, bd1, bd2, bd3)
            for boxIndex in range(boxIndexesCount):
                segmentNodeList = self._boxIndexToSegmentNodeList[boxIndex]
                segmentsParameterLists = []
                for s, n3, n1 in segmentNodeList:
                    segmentsParameterLists.append(
                        self._segments[s].getBoxCoordinatesListAlong(
                            n1, [-2, -1] if self._segmentsIn[s] else [1, 0], n3))
                bx[boxIndex], bd1[boxIndex], bd2[boxIndex], bd3[boxIndex] = \
                    self._sampleMidPoint(segmentsParameterLists)

    def _createBoxBoundaryNodeIdsList(self, s):
        """
        Creates a list (in a circular format similar to other rim node id lists) of core box node ids that are
        located at the boundary of the core. This list is used to easily stitch inner rim nodes with box nodes.
        Used specifically for solid core at the junction.
        :param s: Index for identifying segments.
        :return: A list of box node ids stored in a circular format, and a lookup list that translates indexes used in
        boxBoundaryNodeIds list to indexes that can be used in boxCoordinates list.
        """
        boxBoundaryNodeIds = []
        boxBoundaryNodeToBoxId = []
        elementsCountAcrossMajor = self._segments[s].getElementsCountAcrossMajor()
        elementsCountAcrossMinor = self._segments[s].getElementsCountAcrossMinor()
        elementsCountTransition = self._segments[s].getElementsCountAcrossTransition()
        boxElementsCountRow = (elementsCountAcrossMajor - 2 * elementsCountTransition) + 1
        boxElementsCountColumn = (elementsCountAcrossMinor - 2 * elementsCountTransition) + 1

        for n3 in range(boxElementsCountRow):
            if n3 == 0 or n3 == boxElementsCountRow - 1:
                ids = self._boxNodeIds[s][n3] if n3 == 0 else self._boxNodeIds[s][n3][::-1]
                n1List = list(range(boxElementsCountColumn)) if n3 == 0 else (
                    list(range(boxElementsCountColumn - 1, -1, -1)))
                boxBoundaryNodeIds += [ids[c] for c in range(boxElementsCountColumn)]
                for n1 in n1List:
                    boxBoundaryNodeToBoxId.append([n3, n1])
            else:
                for n1 in [-1, 0]:
                    boxBoundaryNodeIds.append(self._boxNodeIds[s][n3][n1])
                    boxBoundaryNodeToBoxId.append([n3, n1])

        start = elementsCountAcrossMajor - 4 - 2 * (elementsCountTransition - 1)
        idx = elementsCountAcrossMinor - 2 * (elementsCountTransition - 1)
        for n in range(int(start), -1, -1):
            boxBoundaryNodeIds.append(boxBoundaryNodeIds.pop(idx + 2 * n))
            boxBoundaryNodeToBoxId.append(boxBoundaryNodeToBoxId.pop(idx + 2 * n))

        nloop = elementsCountAcrossMinor // 2 - elementsCountTransition
        for _ in range(nloop):
            boxBoundaryNodeIds.insert(len(boxBoundaryNodeIds), boxBoundaryNodeIds.pop(0))
            boxBoundaryNodeToBoxId.insert(len(boxBoundaryNodeToBoxId),
                                              boxBoundaryNodeToBoxId.pop(0))

        return boxBoundaryNodeIds, boxBoundaryNodeToBoxId

    def _getBoxCoordinates(self, n1):
        return (self._boxCoordinates[0][n1], self._boxCoordinates[1][n1],
                self._boxCoordinates[2][n1], self._boxCoordinates[3][n1])

    def _getRimCoordinates(self, n1):
        return (self._rimCoordinates[0][0][n1], self._rimCoordinates[1][0][n1],
                self._rimCoordinates[2][0][n1], self._rimCoordinates[3][0][n1])

    def _generateBoxElements(self, s, n2, mesh, elementtemplate, coordinates, segment, generateData):
        """
        Blackbox function for generating core box elements at a junction.
        """
        annotationMeshGroups = generateData.getAnnotationMeshGroups(segment.getAnnotationTerms())
        boxElementsCountAcrossMinor = self._segments[0].getCoreNodesCountAcrossMinor() - 1
        boxElementsCountAcrossMajor = [self._segments[s].getCoreNodesCountAcrossMajor() - 1
                                       for s in range(self._segmentsCount)]
        acrossMajorCounts = [segment.getElementsCountAcrossMajor() for segment in self._segments]
        is6WayTriplePoint = True if (((max(acrossMajorCounts) - 2) // 2) == (min(acrossMajorCounts) - 2)
                                     and (self._segmentsCount == 3)) else False

        eftList = [[None] * boxElementsCountAcrossMinor for _ in range(boxElementsCountAcrossMajor[s])]
        scalefactorsList = [[None] * boxElementsCountAcrossMinor for _ in range(boxElementsCountAcrossMajor[s])]

        nodeLayout6Way = generateData.getNodeLayout6Way()
        nodeLayout8Way = generateData.getNodeLayout8Way()
        nodeLayoutFlipD2 = generateData.getNodeLayoutFlipD2()
        nodeLayoutBifurcation = generateData.getNodeLayoutBifurcation()

        for e3 in range(boxElementsCountAcrossMajor[s]):
            for e1 in range(boxElementsCountAcrossMinor):
                e3p = (e3 + 1)
                nids, nodeParameters, nodeLayouts = [], [], []
                for n1 in [e1, e1 + 1]:
                    for n3 in [e3, e3p]:
                        nids.append(segment.getBoxNodeIds(n1, n2, n3))
                        boxCoordinates = segment.getBoxCoordinates(n1, n2, n3)
                        nodeParameters.append(boxCoordinates)
                        nodeLayouts.append(None)
                    for n3 in [e3, e3p]:
                        boxIndex = self._segmentNodeToBoxIndex[s][n3][n1]
                        nids.append(self._boxNodeIds[s][n3][n1])
                        nodeParameters.append(self._getBoxCoordinates(boxIndex))
                        segmentNodesCount = len(self._boxIndexToSegmentNodeList[boxIndex])
                        if is6WayTriplePoint and (segmentNodesCount == 3) and self._segmentsCount == 3:
                            nodeLayouts.append(nodeLayoutBifurcation)
                        elif self._segmentsIn[s] and (segmentNodesCount == 3) and self._segmentsCount == 4:
                            location = 1 if e3 < boxElementsCountAcrossMajor[s] // 2 else 2
                            nodeLayoutTrifurcation = generateData.getNodeLayoutTrifurcation(location)
                            nodeLayouts.append(nodeLayout6Way if self._triSequence == [0, 1, 3, 2] else
                                               nodeLayoutTrifurcation)
                        else:
                            nodeLayouts.append(nodeLayoutFlipD2 if (segmentNodesCount == 2) else
                                               nodeLayout6Way if (segmentNodesCount == 3) else
                                               nodeLayout8Way)

                    if not self._segmentsIn[s]:
                        for a in [nids, nodeParameters, nodeLayouts]:
                            a[-4], a[-2] = a[-2], a[-4]
                            a[-3], a[-1] = a[-1], a[-3]
                eft = eftList[e3][e1]
                scalefactors = scalefactorsList[e3][e1]
                if not eft:
                    eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                    eftList[e3][e1] = eft
                    scalefactorsList[e3][e1] = scalefactors
                elementtemplate.defineField(coordinates, -1, eft)
                elementIdentifier = generateData.nextElementIdentifier()
                element = mesh.createElement(elementIdentifier, elementtemplate)
                element.setNodesByIdentifier(eft, nids)
                if scalefactors:
                    element.setScaleFactors(eft, scalefactors)
                for annotationMeshGroup in annotationMeshGroups:
                    annotationMeshGroup.addElement(element)

    def _generateTransitionElements(self, s, n2, mesh, elementtemplate, coordinates, segment, generateData,
                                    elementsCountAround, boxBoundaryNodeIds, boxBoundaryNodeToBoxId):
        """
        Blackbox function for generating core transition elements at a junction.
        """
        annotationMeshGroups = generateData.getAnnotationMeshGroups(segment.getAnnotationTerms())
        nodesCountAcrossMinor = self._segments[0].getCoreNodesCountAcrossMinor()
        nodesCountAcrossMajor = [self._segments[s].getCoreNodesCountAcrossMajor() for s in
                                 range(self._segmentsCount)]
        acrossMajorCounts = [segment.getElementsCountAcrossMajor() for segment in self._segments]
        eftList = [None] * elementsCountAround
        scalefactorsList = [None] * elementsCountAround

        triplePointIndexesList = segment.getTriplePointIndexes()
        is6WayTriplePoint = True if (((max(acrossMajorCounts) - 2) // 2) == (min(acrossMajorCounts) - 2)
                                     and (self._segmentsCount == 3)) else False
        pSegment = acrossMajorCounts.index(max(acrossMajorCounts))
        topMidIndex = (nodesCountAcrossMajor[pSegment] // 2) + (nodesCountAcrossMinor // 2)
        bottomMidIndex = elementsCountAround - topMidIndex
        midIndexes = [topMidIndex, bottomMidIndex]

        nodeLayout6Way = generateData.getNodeLayout6Way()
        nodeLayout8Way = generateData.getNodeLayout8Way()
        nodeLayoutFlipD2 = generateData.getNodeLayoutFlipD2()
        nodeLayoutTransition = generateData.getNodeLayoutTransition()
        nodeLayoutBifurcationTransition = generateData.getNodeLayoutBifurcationTransition()

        for e1 in range(elementsCountAround):
            nids, nodeParameters, nodeLayouts = [], [], []
            n1p = (e1 + 1) % elementsCountAround
            oLocation = segment.getTriplePointLocation(e1)
            for n3 in range(2):
                if n3 == 0:  # core box region
                    for n1 in [e1, n1p]:
                        nids.append(segment.getBoxBoundaryNodeIds(n1, n2))
                        n3c, n1c = segment.getBoxBoundaryNodeToBoxId(n1, n2)
                        nodeParameters.append(segment.getBoxCoordinates(n1c, n2, n3c))
                        nodeLayoutTransitionTriplePoint = (
                            generateData.getNodeLayoutTransitionTriplePoint(oLocation))
                        nodeLayouts.append(nodeLayoutTransitionTriplePoint if n1 in triplePointIndexesList
                                           else nodeLayoutTransition)
                    for n1 in [e1, n1p]:
                        nids.append(boxBoundaryNodeIds[n1])
                        n3c, n1c = boxBoundaryNodeToBoxId[n1]
                        boxIndex = self._segmentNodeToBoxIndex[s][n3c][n1c]
                        nodeParameters.append(self._getBoxCoordinates(boxIndex))
                        segmentNodesCount = len(self._boxIndexToSegmentNodeList[boxIndex])
                        if segmentNodesCount == 3:  # 6-way node
                            if is6WayTriplePoint: # Special 6-way triple point case
                                if (n1 in triplePointIndexesList or (s == pSegment and n1 in midIndexes)):
                                    # 6-way AND triple-point node - only applies to bifurcations
                                    location = (midIndexes.index(n1) + 1) if oLocation == 0 else oLocation
                                    if (s == 1 and n1 == n1p) or (s == 2 and n1 == e1):
                                        location = 3 if abs(location) == 1 else location
                                    elif (s == 1 and n1 != n1p) or (s == 2 and n1 != e1):
                                        location = 4 if abs(location) == 2 else location
                                    nodeLayout = generateData.getNodeLayout6WayTriplePoint(location)
                                else:
                                    nodeLayout = nodeLayoutBifurcationTransition
                            elif self._segmentsCount == 4 and self._segmentsIn[s]: # Trifurcation case
                                location = \
                                    1 if (e1 < elementsCountAround // 4) or (e1 >= 3 * elementsCountAround // 4) else 2
                                nodeLayoutTrifurcation = generateData.getNodeLayoutTrifurcation(location)
                                nodeLayout = nodeLayout6Way if self._triSequence == [0, 1, 3, 2] else (
                                    nodeLayoutTrifurcation)
                            else:
                                nodeLayout = nodeLayout6Way
                        elif segmentNodesCount == 4:  # 8-way node
                            nodeLayout = nodeLayout8Way
                        elif n1 in triplePointIndexesList:  # triple-point node
                            location = oLocation
                            if self._segmentsCount == 3: # bifurcation
                                sequence = self._biSequence
                                condition1 = oLocation > 0
                                condition2 = oLocation < 0
                                if self._segmentsIn.count(True) == 0:
                                    condition = condition1 if sequence == [0, 2, 1] else condition2
                                    location *= -1 if s == 2 or (s == 1 and condition) else 1
                                elif self._segmentsIn.count(True) == 1:
                                    condition = condition1 if sequence == [0, 2, 1] else condition2
                                    location *= -1 if s == 2 and condition else 1
                                elif self._segmentsIn.count(True) == 2:
                                    condition = condition2 if sequence == [0, 2, 1] else condition1
                                    location *= -1 if s == 1 and condition else 1
                                elif self._segmentsIn.count(True) == 3:
                                    condition = condition2 if sequence == [0, 2, 1] else condition1
                                    location *= -1 if (s == 1 and condition) or (s == 2) else 1
                            elif self._segmentsCount == 4: # trifurcation
                                sequence = self._triSequence
                                s0 = (s - 1) % self._segmentsCount
                                s1 = (s + 1) % self._segmentsCount
                                if sequence == [0, 1, 3, 2]:
                                    if self._segmentsIn == [True, False, False, False] and self._segmentsIn[s1]:
                                        location = (oLocation) * -1
                                    elif self._segmentsIn == [True, True, False, False] and \
                                        self._segmentsIn[s] != self._segmentsIn[s1]:
                                        location = abs(oLocation)
                                elif sequence == [0, 1, 2, 3] and \
                                        (self._segmentsIn[s1] or all(not self._segmentsIn[n] for n in [s, s0, s1])):
                                    location = abs(oLocation)
                            nodeLayout = generateData.getNodeLayoutTransitionTriplePoint(location)
                        else:
                            nodeLayout = nodeLayoutTransition
                        nodeLayouts.append(nodeLayout)

                    if not self._segmentsIn[s]:
                        for a in [nids, nodeParameters, nodeLayouts]:
                            a[-4], a[-2] = a[-2], a[-4]
                            a[-3], a[-1] = a[-1], a[-3]
            else:  # rim region
                for n1 in [e1, n1p]:
                    nids.append(segment.getRimNodeId(n1, n2, 0))
                    nodeParameters.append(segment.getRimCoordinates(n1, n2, 0))
                    nodeLayouts.append(None)
                for n1 in [e1, n1p]:
                    rimIndex = self._segmentNodeToRimIndex[s][n1]
                    nids.append(self._rimNodeIds[0][rimIndex])
                    nodeParameters.append(self._getRimCoordinates(rimIndex))
                    segmentNodesCount = len(self._rimIndexToSegmentNodeList[rimIndex])
                    nodeLayouts.append(nodeLayoutFlipD2 if (segmentNodesCount == 2) else
                                       nodeLayout6Way if (segmentNodesCount == 3) else
                                       nodeLayout8Way)
                if not self._segmentsIn[s]:
                    for a in [nids, nodeParameters, nodeLayouts]:
                        a[-4], a[-2] = a[-2], a[-4]
                        a[-3], a[-1] = a[-1], a[-3]
            eft = eftList[e1]
            scalefactors = scalefactorsList[e1]
            if not eft:
                eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                eftList[e1] = eft
                scalefactorsList[e1] = scalefactors
            elementtemplate.defineField(coordinates, -1, eft)
            elementIdentifier = generateData.nextElementIdentifier()
            element = mesh.createElement(elementIdentifier, elementtemplate)
            element.setNodesByIdentifier(eft, nids)
            if scalefactors:
                element.setScaleFactors(eft, scalefactors)
            for annotationMeshGroup in annotationMeshGroups:
                annotationMeshGroup.addElement(element)

    def generateMesh(self, generateData: TubeNetworkMeshGenerateData):
        if generateData.isShowTrimSurfaces():
            dimension = generateData.getMeshDimension()
            nodeIdentifier, elementIdentifier = generateData.getNodeElementIdentifiers()
            faceIdentifier = elementIdentifier if (dimension == 2) else None
            for s in range(self._segmentsCount):
                for trimSurface in self._trimSurfaces[s]:
                    if trimSurface:
                        nodeIdentifier, faceIdentifier = \
                            trimSurface.generateMesh(generateData.getRegion(), nodeIdentifier, faceIdentifier)
            if dimension == 2:
                elementIdentifier = faceIdentifier
            generateData.setNodeElementIdentifiers(nodeIdentifier, elementIdentifier)

        if self._segmentsCount < 3:
            return

        rimIndexesCount = len(self._rimIndexToSegmentNodeList)
        elementsCountTransition = self._segments[0].getElementsCountTransition()
        nodesCountRim = self._segments[0].getNodesCountRim() + (elementsCountTransition - 1) if self._isCore else (
            self._segments[0].getNodesCountRim())
        elementsCountRim = max(1, nodesCountRim - 1)
        if self._rimCoordinates:
            self._rimNodeIds = [[None] * rimIndexesCount for _ in range(nodesCountRim)]

        if self._boxCoordinates:
            nodesCountAcrossMinor = self._segments[0].getCoreNodesCountAcrossMinor()
            acrossMajorCounts = [segment.getElementsCountAcrossMajor() for segment in self._segments]
            self._boxNodeIds = [[[None for _ in range(nodesCountAcrossMinor)] for _ in range(acrossMajorCounts[s])]
                                for s in range(self._segmentsCount)]

        coordinates = generateData.getCoordinates()
        fieldcache = generateData.getFieldcache()
        nodes = generateData.getNodes()
        nodetemplate = generateData.getNodetemplate()
        isLinearThroughWall = generateData.isLinearThroughWall()
        mesh = generateData.getMesh()
        meshDimension = generateData.getMeshDimension()
        elementtemplate = mesh.createElementtemplate()
        elementtemplate.setElementShapeType(
            Element.SHAPE_TYPE_CUBE if (meshDimension == 3) else Element.SHAPE_TYPE_SQUARE)
        d3Defined = (meshDimension == 3) and not isLinearThroughWall

        nodeLayout6Way = generateData.getNodeLayout6Way()
        nodeLayout8Way = generateData.getNodeLayout8Way()
        nodeLayoutFlipD2 = generateData.getNodeLayoutFlipD2()

        # nodes and elements are generated in order of segments
        for s in range(self._segmentsCount):
            segment = self._segments[s]
            elementsCountAlong = segment.getSampledElementsCountAlong()
            e2 = (elementsCountAlong - 1) if self._segmentsIn[s] else 0
            n2 = (elementsCountAlong - 1) if self._segmentsIn[s] else 1
            segment.generateMesh(generateData, n2Only=n2)

            elementsCountAround = segment.getElementsCountAround()

            # Create nodes
            if self._boxCoordinates:
                # create box nodes
                bx, bd1, bd2, bd3 = (self._boxCoordinates[0], self._boxCoordinates[1],
                                     self._boxCoordinates[2], self._boxCoordinates[3])
                for n3 in range(acrossMajorCounts[s] - 1):
                    for n1 in range(nodesCountAcrossMinor):
                        boxIndex = self._segmentNodeToBoxIndex[s][n3][n1]
                        segmentNodeList = self._boxIndexToSegmentNodeList[boxIndex]
                        nodeIdentifiersCheck = []
                        for segmentNodes in segmentNodeList:
                            sp, n3p, n1p = segmentNodes
                            nodeIdentifiersCheck.append(self._boxNodeIds[sp][n3p][n1p])
                        if nodeIdentifiersCheck.count(None) == len(nodeIdentifiersCheck):
                            nodeIdentifier = generateData.nextNodeIdentifier()
                            node = nodes.createNode(nodeIdentifier, nodetemplate)
                            fieldcache.setNode(node)

                            for nodeValue, bValue in zip([Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                                                          Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3],
                                                         [bx, bd1, bd2, bd3]):
                                coordinates.setNodeParameters(fieldcache, -1, nodeValue, 1, bValue[boxIndex])
                        else:
                            nodeIdentifier = next(id for id in nodeIdentifiersCheck if id is not None)

                        self._boxNodeIds[s][n3][n1] = nodeIdentifier

                boxBoundaryNodeIds, boxBoundaryNodeToBoxId = self._createBoxBoundaryNodeIdsList(s)

            if self._rimCoordinates:
                # create rim nodes (including core transition nodes)
                for n3 in range(nodesCountRim):
                    rx = self._rimCoordinates[0][n3]
                    rd1 = self._rimCoordinates[1][n3]
                    rd2 = self._rimCoordinates[2][n3]
                    rd3 = self._rimCoordinates[3][n3] if d3Defined else None
                    layerNodeIds = self._rimNodeIds[n3]
                    for n1 in range(elementsCountAround):
                        rimIndex = self._segmentNodeToRimIndex[s][n1]
                        nodeIdentifier = self._rimNodeIds[n3][rimIndex]
                        if nodeIdentifier is not None:
                            continue
                        nodeIdentifier = generateData.nextNodeIdentifier()
                        node = nodes.createNode(nodeIdentifier, nodetemplate)
                        fieldcache.setNode(node)
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, rx[rimIndex])
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, rd1[rimIndex])
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, rd2[rimIndex])
                        if rd3:
                            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, rd3[rimIndex])
                        layerNodeIds[rimIndex] = nodeIdentifier

            # Create elements
            if self._isCore:
                # create box elements
                self._generateBoxElements(s, n2, mesh, elementtemplate, coordinates, segment, generateData)
                # create core transition elements
                self._generateTransitionElements(s, n2, mesh, elementtemplate, coordinates, segment, generateData,
                    elementsCountAround, boxBoundaryNodeIds, boxBoundaryNodeToBoxId)

            if self._rimCoordinates:
                # create rim elements
                annotationMeshGroups = generateData.getAnnotationMeshGroups(segment.getAnnotationTerms())
                eftList = [None] * elementsCountAround
                scalefactorsList = [None] * elementsCountAround
                for e3 in range(elementsCountRim):
                    for e1 in range(elementsCountAround):
                        n1p = (e1 + 1) % elementsCountAround
                        nids = []
                        nodeParameters = []
                        nodeLayouts = []
                        for n3 in [e3, e3 + 1] if (meshDimension == 3) else [e3]:
                            for n1 in [e1, n1p]:
                                nids.append(self._segments[s].getRimNodeId(n1, n2, n3))
                                if e3 == 0:
                                    rimCoordinates = self._segments[s].getRimCoordinates(n1, n2, n3)
                                    nodeParameters.append(rimCoordinates if d3Defined else
                                                          (rimCoordinates[0], rimCoordinates[1], rimCoordinates[2],
                                                           None))
                                    nodeLayouts.append(None)
                            for n1 in [e1, n1p]:
                                rimIndex = self._segmentNodeToRimIndex[s][n1]
                                nids.append(self._rimNodeIds[n3][rimIndex])
                                if e3 == 0:
                                    nodeParameters.append(
                                        (self._rimCoordinates[0][n3][rimIndex],
                                         self._rimCoordinates[1][n3][rimIndex],
                                         self._rimCoordinates[2][n3][rimIndex],
                                         self._rimCoordinates[3][n3][rimIndex] if d3Defined else None))
                                    segmentNodesCount = len(self._rimIndexToSegmentNodeList[rimIndex])
                                    nodeLayouts.append(nodeLayoutFlipD2 if (segmentNodesCount == 2) else
                                                       nodeLayout6Way if (segmentNodesCount == 3) else
                                                       nodeLayout8Way)
                            if not self._segmentsIn[s]:
                                for a in [nids, nodeParameters, nodeLayouts] if (e3 == 0) else [nids]:
                                    a[-4], a[-2] = a[-2], a[-4]
                                    a[-3], a[-1] = a[-1], a[-3]
                        # exploit efts being same through the wall
                        eft = eftList[e1]
                        scalefactors = scalefactorsList[e1]
                        if not eft:
                            eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                            eftList[e1] = eft
                            scalefactorsList[e1] = scalefactors
                        elementtemplate.defineField(coordinates, -1, eft)
                        elementIdentifier = generateData.nextElementIdentifier()
                        element = mesh.createElement(elementIdentifier, elementtemplate)
                        segment.setRimElementId(e1, e2, e3, elementIdentifier)
                        element.setNodesByIdentifier(eft, nids)
                        if scalefactors:
                            element.setScaleFactors(eft, scalefactors)
                        for annotationMeshGroup in annotationMeshGroups:
                            annotationMeshGroup.addElement(element)


class TubeNetworkMeshBuilder(NetworkMeshBuilder):

    def __init__(self, networkMesh: NetworkMesh, targetElementDensityAlongLongestSegment: float,
                 defaultElementsCountAround: int, elementsCountThroughWall: int,
                 layoutAnnotationGroups: list = [], annotationElementsCountsAround: list = [],
                 defaultElementsCountAcrossMajor: int = 4, elementsCountTransition: int = 1,
                 annotationElementsCountsAcrossMajor: list = [], isCore=False):
        super(TubeNetworkMeshBuilder, self).__init__(
            networkMesh, targetElementDensityAlongLongestSegment, layoutAnnotationGroups)
        self._defaultElementsCountAround = defaultElementsCountAround
        self._elementsCountThroughWall = elementsCountThroughWall
        self._layoutAnnotationGroups = layoutAnnotationGroups
        self._annotationElementsCountsAround = annotationElementsCountsAround
        layoutFieldmodule = self._layoutRegion.getFieldmodule()
        self._layoutInnerCoordinates = layoutFieldmodule.findFieldByName("inner coordinates").castFiniteElement()
        if not self._layoutInnerCoordinates.isValid():
            self._layoutInnerCoordinates = None
        self._isCore = isCore
        self._defaultElementsCountAcrossMajor = defaultElementsCountAcrossMajor
        self._elementsCountTransition = elementsCountTransition
        self._annotationElementsCountsAcrossMajor = annotationElementsCountsAcrossMajor

    def createSegment(self, networkSegment):
        pathParametersList = [get_nodeset_path_ordered_field_parameters(
            self._layoutNodes, self._layoutCoordinates, pathValueLabels,
            networkSegment.getNodeIdentifiers(), networkSegment.getNodeVersions())]
        if self._layoutInnerCoordinates:
            pathParametersList.append(get_nodeset_path_ordered_field_parameters(
                self._layoutNodes, self._layoutInnerCoordinates, pathValueLabels,
                networkSegment.getNodeIdentifiers(), networkSegment.getNodeVersions()))
        elementsCountAround = self._defaultElementsCountAround
        elementsCountAcrossMajor = self._defaultElementsCountAcrossMajor
        elementsCountAcrossMinor = (((elementsCountAround - 4) // 4 - elementsCountAcrossMajor // 2) * 2 + 6)

        i = 0
        for layoutAnnotationGroup in self._layoutAnnotationGroups:
            if i >= len(self._annotationElementsCountsAround):
                break
            if self._annotationElementsCountsAround[i] > 0:
                if networkSegment.hasLayoutElementsInMeshGroup(layoutAnnotationGroup.getMeshGroup(self._layoutMesh)):
                    elementsCountAround = self._annotationElementsCountsAround[i]
                    break
            i += 1
        if self._isCore:
            annotationElementsCountAcrossMinor = []
            i = 0
            for layoutAnnotationGroup in self._layoutAnnotationGroups:
                if i >= len(self._annotationElementsCountsAcrossMajor):
                    break
                if self._annotationElementsCountsAcrossMajor[i] > 0:
                    if networkSegment.hasLayoutElementsInMeshGroup(
                            layoutAnnotationGroup.getMeshGroup(self._layoutMesh)):
                        elementsCountAcrossMajor = self._annotationElementsCountsAcrossMajor[i]
                        elementsCountAcrossMinor = (
                                ((elementsCountAround - 4) // 4 - elementsCountAcrossMajor // 2) * 2 + 6)
                        annotationElementsCountAcrossMinor.append(elementsCountAcrossMinor)
                        break
                i += 1
            # elements count across minor must be the same for all annotation groups
            assert all(value == annotationElementsCountAcrossMinor[0] for value in annotationElementsCountAcrossMinor)

        return TubeNetworkMeshSegment(networkSegment, pathParametersList, elementsCountAround,
                                      self._elementsCountThroughWall, self._isCore, elementsCountAcrossMajor,
                                      elementsCountAcrossMinor, self._elementsCountTransition)

    def createJunction(self, inSegments, outSegments):
        """
        :param inSegments: List of inward TubeNetworkMeshSegment.
        :param outSegments: List of outward TubeNetworkMeshSegment.
        :return: A TubeNetworkMeshJunction.
        """
        return TubeNetworkMeshJunction(inSegments, outSegments)


def getPathRawTubeCoordinates(pathParameters, elementsCountAround, radius=1.0, phaseAngle=0.0):
    """
    Generate coordinates around and along a tube in parametric space around the path parameters,
    at xi2^2 + xi3^2 = radius at the same density as path parameters.
    :param pathParameters: List over nodes of 6 parameters vectors [cx, cd1, cd2, cd12, cd3, cd13] giving
    coordinates cx along path centre, derivatives cd1 along path, cd2 and cd3 giving side vectors,
    and cd12, cd13 giving rate of change of side vectors. Parameters have 3 components.
    Same format as output of zinc_utils get_nodeset_path_ordered_field_parameters().
    :param elementsCountAround: Number of elements & nodes to create around tube. First location is at +d2.
    :param radius: Radius of tube in xi space.
    :param phaseAngle: Starting angle around ellipse, where 0.0 is at d2, pi/2 is at d3.
    :return: px[][], pd1[][], pd2[][], pd12[][] with first index in range(pointsCountAlong),
    second inner index in range(elementsCountAround)
    """
    assert len(pathParameters) == 6
    pointsCountAlong = len(pathParameters[0])
    assert pointsCountAlong > 1
    assert len(pathParameters[0][0]) == 3

    # sample around circle in xi space, later smooth and re-sample to get even spacing in geometric space
    ellipsePointCount = 16
    aroundScale = 2.0 * math.pi / ellipsePointCount
    sxi = []
    sdxi = []
    angleBetweenPoints = 2.0 * math.pi / ellipsePointCount
    for q in range(ellipsePointCount):
        theta = phaseAngle + q * angleBetweenPoints
        xi2 = radius * math.cos(theta)
        xi3 = radius * math.sin(theta)
        sxi.append([xi2, xi3])
        dxi2 = -xi3 * aroundScale
        dxi3 = xi2 * aroundScale
        sdxi.append([dxi2, dxi3])

    px = []
    pd1 = []
    pd2 = []
    pd12 = []
    for p in range(pointsCountAlong):
        cx, cd1, cd2, cd12, cd3, cd13 = [cp[p] for cp in pathParameters]
        tx = []
        td1 = []
        for q in range(ellipsePointCount):
            xi2 = sxi[q][0]
            xi3 = sxi[q][1]
            x = [(cx[c] + xi2 * cd2[c] + xi3 * cd3[c]) for c in range(3)]
            tx.append(x)
            dxi2 = sdxi[q][0]
            dxi3 = sdxi[q][1]
            d1 = [(dxi2 * cd2[c] + dxi3 * cd3[c]) for c in range(3)]
            td1.append(d1)
        # smooth to get reasonable derivative magnitudes
        td1 = smoothCubicHermiteDerivativesLoop(tx, td1, fixAllDirections=True)
        # resample to get evenly spaced points around loop, temporarily adding start point to end
        ex, ed1, pe, pxi, psf = sampleCubicHermiteCurvesSmooth(tx + tx[:1], td1 + td1[:1], elementsCountAround)
        exi, edxi = interpolateSampleCubicHermite(sxi + sxi[:1], sdxi + sdxi[:1], pe, pxi, psf)
        ex.pop()
        ed1.pop()
        exi.pop()
        edxi.pop()

        # check closeness of x(exi[i]) to ex[i]
        # find nearest xi2, xi3 if above is finite error
        # a small but non-negligible error, but results look fine so not worrying
        # dxi = []
        # for i in range(len(ex)):
        #     xi2 = exi[i][0]
        #     xi3 = exi[i][1]
        #     xi23 = xi2 * xi3
        #     x = [(cx[c] + xi2 * cd2[c] + xi3 * cd3[c]) for c in range(3)]
        #     dxi.append(sub(x, ex[i]))
        # print("error", p, "=", [magnitude(v) for v in dxi])

        # calculate d2, d12 at exi
        ed2 = []
        ed12 = []
        for i in range(len(ex)):
            xi2 = exi[i][0]
            xi3 = exi[i][1]
            d2 = [(cd1[c] + xi2 * cd12[c] + xi3 * cd13[c]) for c in range(3)]
            ed2.append(d2)
            dxi2 = edxi[i][0]
            dxi3 = edxi[i][1]
            d12 = [(dxi2 * cd12[c] + dxi3 * cd13[c]) for c in range(3)]
            ed12.append(d12)

        px.append(ex)
        pd1.append(ed1)
        pd2.append(ed2)
        pd12.append(ed12)

    return px, pd1, pd2, pd12


def resampleTubeCoordinates(rawTubeCoordinates, fixedElementsCountAlong=None,
                            targetElementLength=None, minimumElementsCountAlong=1,
                            startSurface: TrackSurface=None, endSurface: TrackSurface=None):
    """
    Generate new tube coordinates along raw tube coordinates, optionally trimmed to start/end surfaces.
    Untrimmed tube elements are even sized along each longitudinal curve.
    Trimmed tube elements adjust derivatives at trimmed ends to transition from distorted to regular spacing.
    Can specify either fixedElementsCountAlong or targetElementLength.
    :param rawTubeCoordinates: (px, pd1, pd2, pd12) returned by getPathRawTubeCoordinates().
    :param fixedElementsCountAlong: Number of elements in resampled coordinates, or None to use targetElementLength.
    :param targetElementLength: Target element length or None to use fixedElementsCountAlong.
    Length is compared with mean trimmed length to determine number along, subject to specified minimum.
    :param minimumElementsCountAlong: Minimum number along when targetElementLength is used.
    :param startSurface: Optional TrackSurface specifying start of tube at intersection with it.
    :param endSurface: Optional TrackSurface specifying end of tube at intersection with it.
    :return: sx[][], sd1[][], sd2[][], sd12[][] with first index in range(elementsCountAlong + 1),
    second inner index in range(elementsCountAround)
    """
    assert fixedElementsCountAlong or targetElementLength
    px, pd1, pd2, pd12 = rawTubeCoordinates
    pointsCountAlong = len(px)
    endPointLocation = float(pointsCountAlong - 1)
    elementsCountAround = len(px[0])

    # work out lengths of longitudinal curves, raw and trimmed
    sumLengths = 0.0
    startCurveLocations = []
    startLengths = []
    meanStartLocation = 0.0
    endCurveLocations = []
    endLengths = []
    meanEndLocation = 0.0
    for q in range(elementsCountAround):
        cx = [px[p][q] for p in range(pointsCountAlong)]
        cd2 = [pd2[p][q] for p in range(pointsCountAlong)]
        startCurveLocation = None
        if startSurface:
            startSurfacePosition, startCurveLocation, startIntersects = startSurface.findNearestPositionOnCurve(cx, cd2)
            if startIntersects:
                meanStartLocation += startCurveLocation[0] + startCurveLocation[1]
            else:
                startCurveLocation = None
        startCurveLocations.append(startCurveLocation)
        endCurveLocation = None
        if endSurface:
            endSurfacePosition, endCurveLocation, endIntersects = endSurface.findNearestPositionOnCurve(cx, cd2)
            if endIntersects:
                meanEndLocation += endCurveLocation[0] + endCurveLocation[1]
            else:
                endCurveLocation = None
        if not endCurveLocation:
            meanEndLocation += endPointLocation
        endCurveLocations.append(endCurveLocation)
        startLength, length, endLength = \
            getCubicHermiteTrimmedCurvesLengths(cx, cd2, startCurveLocation, endCurveLocation)[0:3]
        sumLengths += length
        startLengths.append(startLength)
        endLengths.append(endLength)

    meanLength = sumLengths / elementsCountAround
    if fixedElementsCountAlong:
        elementsCountAlong = fixedElementsCountAlong
    else:
        # small fudge factor so whole numbers chosen on centroid don't go one higher:
        elementsCountAlong = max(minimumElementsCountAlong, math.ceil(meanLength * 0.999 / targetElementLength))
    meanStartLocation /= elementsCountAround
    e = min(int(meanStartLocation), pointsCountAlong - 2)
    meanStartCurveLocation = (e, meanStartLocation - e)
    meanEndLocation /= elementsCountAround
    e = min(int(meanEndLocation), pointsCountAlong - 2)
    meanEndCurveLocation = (e, meanEndLocation - e)

    # resample along, with variable spacing where ends are trimmed
    sx = [[None] * elementsCountAround for _ in range(elementsCountAlong + 1)]
    sd1 = [[None] * elementsCountAround for _ in range(elementsCountAlong + 1)]
    sd2 = [[None] * elementsCountAround for _ in range(elementsCountAlong + 1)]
    sd12 = [[None] * elementsCountAround for _ in range(elementsCountAlong + 1)]
    for q in range(elementsCountAround):
        cx = [px[p][q] for p in range(pointsCountAlong)]
        cd1 = [pd1[p][q] for p in range(pointsCountAlong)]
        cd2 = [pd2[p][q] for p in range(pointsCountAlong)]
        cd12 = [pd12[p][q] for p in range(pointsCountAlong)]
        meanStartLength, meanLength, meanEndLength = \
            getCubicHermiteTrimmedCurvesLengths(cx, cd2, meanStartCurveLocation, meanEndCurveLocation)[0:3]
        derivativeMagnitudeStart = (meanLength + 2.0 * (meanStartLength - startLengths[q])) / elementsCountAlong
        derivativeMagnitudeEnd = (meanLength + 2.0 * (meanEndLength - endLengths[q])) / elementsCountAlong
        qx, qd2, pe, pxi, psf = sampleCubicHermiteCurvesSmooth(
            cx, cd2, elementsCountAlong, derivativeMagnitudeStart, derivativeMagnitudeEnd,
            startLocation=startCurveLocations[q], endLocation=endCurveLocations[q])
        qd1, qd12 = interpolateSampleCubicHermite(cd1, cd12, pe, pxi, psf)
        # swizzle
        for p in range(elementsCountAlong + 1):
            sx[p][q] = qx[p]
            sd1[p][q] = qd1[p]
            sd2[p][q] = qd2[p]
            sd12[p][q] = qd12[p]

    # recalculate d1 around intermediate rings, but still in plane
    # normally looks fine, but d1 derivatives are wavy when very distorted
    pStart = 0 if startSurface else 1
    pLimit = elementsCountAlong + 1 if endSurface else elementsCountAlong
    for p in range(pStart, pLimit):
        # first smooth to get d1 with new directions not tangential to surface
        td1 = smoothCubicHermiteDerivativesLoop(sx[p], sd1[p])
        # constraint to be tangential to surface
        td1 = [rejection(td1[q], normalize(cross(sd1[p][q], sd2[p][q]))) for q in range(elementsCountAround)]
        # smooth magnitudes only
        sd1[p] = smoothCubicHermiteDerivativesLoop(sx[p], td1, fixAllDirections=True)

    return sx, sd1, sd2, sd12
