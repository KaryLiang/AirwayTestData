
import os, vtk, time
import pydicom
import numpy as np
# from AirwayModule import *
from skimage.measure import label
from skimage import transform
from scipy import ndimage
from vtkplotter import *
from vmtk import vtkvmtk
import matplotlib
matplotlib.use('qt5agg')
import numba as nb
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy
import itk

if __name__ == '__main__':

    d = np.load('.\keypoints\data\PA000019.npz', allow_pickle=True)
    mask = d['isoMask']
    # mask = ndimage.interpolation.zoom(mask, zoom=2, order=0)
    # dilate
    maskITK = itk.GetImageFromArray(mask)
    PixelType = itk.UC
    Dimension = 3
    radiusValue = 3 # 8
    foregroundValue = np.max(mask)
    ImageType = itk.Image[PixelType, Dimension]
    StructuringElementType = itk.FlatStructuringElement[Dimension]
    structuringElement = StructuringElementType.Ball(radiusValue)
    DilateFilterType = itk.BinaryDilateImageFilter[ImageType,
                                                   ImageType,
                                                   StructuringElementType]
    dilateFilter = DilateFilterType.New()
    dilateFilter.SetInput(maskITK)
    dilateFilter.SetKernel(structuringElement)
    dilateFilter.SetForegroundValue(1) # Value to dilate

    lines = d['skeleton']
    source = np.flip(np.array(lines[0]['endpoint1']),0)

    # img = Volume(mask)
    img = Volume(itk.GetArrayFromImage(dilateFilter.GetOutput()))
    surface = img.isosurface(1)
    # show(surface)
    surface = surface._polydata

    surfaceCleaner = vtk.vtkCleanPolyData()
    surfaceCleaner.SetInputData(surface)
    surfaceCleaner.Update()

    surfaceTriangulator = vtk.vtkTriangleFilter()
    surfaceTriangulator.SetInputConnection(surfaceCleaner.GetOutputPort())
    surfaceTriangulator.PassLinesOff()
    surfaceTriangulator.PassVertsOff()
    surfaceTriangulator.Update()

    centerlineInputSurface = surfaceTriangulator.GetOutput()

    surfaceCapper = vtkvmtk.vtkvmtkCapPolyData()
    surfaceCapper.SetInputConnection(surfaceTriangulator.GetOutputPort())
    surfaceCapper.SetDisplacement(0.0)
    surfaceCapper.SetInPlaneDisplacement(0.0)
    surfaceCapper.Update()
    centerlineInputSurface = surfaceCapper.GetOutput()
    capCenterIds = surfaceCapper.GetCapCenterIds()

    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(centerlineInputSurface)
    pointLocator.BuildLocator()


    sourceId = vtk.vtkIdList()
    idsource = pointLocator.FindClosestPoint(source)
    sourceId.InsertNextId(idsource)

    targetId = vtk.vtkIdList()
    allTargetId = []
    allSourceId = []
    for idx, line in enumerate(lines):

        if line['p2IsBif'] == 0:
            idtarget = pointLocator.FindClosestPoint(np.flip(np.array(line['endpoint2']), 0))
            allTargetId.append(idtarget)
            targetId.InsertNextId(idtarget)
            print(lines[0]['endpoint1'], idsource, line['endpoint2'], idtarget)


    start = time.time()
    centerlineFilter = vtkvmtk.vtkvmtkPolyDataCenterlines()
    centerlineFilter.SetInputData(centerlineInputSurface)
    centerlineFilter.SetSourceSeedIds(sourceId)
    centerlineFilter.SetTargetSeedIds(targetId)
    centerlineFilter.SetRadiusArrayName('MaximumInscribedSphereRadius')
    centerlineFilter.SetCostFunction('1/R')
    centerlineFilter.SetFlipNormals(0)
    centerlineFilter.SetAppendEndPointsToCenterlines(0)
    centerlineFilter.SetSimplifyVoronoi(0)
    centerlineFilter.SetCenterlineResampling(0)
    centerlineFilter.SetResamplingStepLength(1.0)
    centerlineFilter.Update()
    end = time.time()
    print(end-start)
    Centerlines = centerlineFilter.GetOutput()
    centerLinePoints = Centerlines.GetPoints()
    voronoiDiagram = centerlineFilter.GetVoronoiDiagram()

    voronoiPoints = vtk_to_numpy(voronoiDiagram.GetPoints().GetData())
    voronoiRadius = vtk_to_numpy(voronoiDiagram.GetPointData().GetArray('MaximumInscribedSphereRadius'))

    lines = Centerlines.GetLines()
    numberOfLines = lines.GetNumberOfCells()

    points = Centerlines.GetPoints()
    lines.InitTraversal()
    idList = vtk.vtkIdList()
    linesList = []
    while lines.GetNextCell(idList):
        n = idList.GetNumberOfIds()
        lineList = []
        for id in range(n):
            # print(idList.GetId(id))
            pointId = idList.GetId(id)
            point = points.GetPoint(pointId)
            lineList.append(list(point))
        linesList.append(lineList)

    vtkLines = []
    for lineList in linesList:
        if len(lineList) == 2:
            continue
        startPoints = []
        for idx, point in enumerate(lineList):
            if idx == 0:
                continue
            startPoints.append([lineList[idx-1], lineList[idx]])
        vtkLine = Lines(startPoints, c='red')
        vtkLines.append(vtkLine)
    # show(img, vtkLines)
    # mask = transform.resize(mask, [mask.shape[0]*2, mask.shape[1]*2, mask.shape[2]*2])
    img = Volume(mask)
    # img = Volume(itk.GetArrayFromImage(dilateFilter.GetOutput()))
    surface = img.isosurface(1)
    # show(surface)
    surface._polydata = centerlineInputSurface
    show(surface.alpha(0.2).color('blue'), vtkLines)
    #
