
#--------------------------------------------------------------

# Global timestep output options
timeStepToStartOutputAt=0
forceOutputAtFirstCall=False

# Global screenshot output options
imageFileNamePadding=0
rescale_lookuptable=True

# Whether or not to request specific arrays from the adaptor.
requestSpecificArrays=False

# a root directory under which all Catalyst output goes
rootDirectory=''

# makes a cinema D index table
make_cinema_table=False

#--------------------------------------------------------------
# Code generated from cpstate.py to create the CoProcessor.
# paraview version 5.6.0
#--------------------------------------------------------------

from paraview.simple import *
from paraview import coprocessing

# ----------------------- CoProcessor definition -----------------------

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:
      # state file generated using paraview version 5.6.0

      # ----------------------------------------------------------------
      # setup views used in the visualization
      # ----------------------------------------------------------------

      # trace generated using paraview version 5.6.0
      #
      # To ensure correct image size when batch processing, please search 
      # for and uncomment the line `# renderView*.ViewSize = [*,*]`

      #### disable automatic camera reset on 'Show'
      paraview.simple._DisableFirstRenderCameraReset()

      # Create a new 'Render View'
      renderView1 = CreateView('RenderView')
      renderView1.ViewSize = [1338, 737]
      renderView1.AxesGrid = 'GridAxes3DActor'
      renderView1.CenterOfRotation = [0.5, 375.0, -125.0]
      renderView1.StereoType = 0
      renderView1.CameraPosition = [1527.764052736578, 375.0, -125.0]
      renderView1.CameraFocalPoint = [0.5, 375.0, -125.0]
      renderView1.CameraViewUp = [0.0, 0.0, 1.0]
      renderView1.CameraParallelScale = 395.28502374868697
      renderView1.Background = [0.32, 0.34, 0.43]

      # init the 'GridAxes3DActor' selected for 'AxesGrid'
      renderView1.AxesGrid.XTitleFontFile = ''
      renderView1.AxesGrid.YTitleFontFile = ''
      renderView1.AxesGrid.ZTitleFontFile = ''
      renderView1.AxesGrid.XLabelFontFile = ''
      renderView1.AxesGrid.YLabelFontFile = ''
      renderView1.AxesGrid.ZLabelFontFile = ''

      # register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView1,
          filename='image_%t.png', freq=1, fittoscreen=0, magnification=1, width=1338, height=737, cinema={})
      renderView1.ViewTime = datadescription.GetTime()

      # ----------------------------------------------------------------
      # restore active view
      SetActiveView(renderView1)
      # ----------------------------------------------------------------

      # ----------------------------------------------------------------
      # setup the data processing pipelines
      # ----------------------------------------------------------------

      # create a new 'OpenFOAMReader'
      # create a producer from a simulation input
      case1OpenFOAM = coprocessor.CreateProducer(datadescription, 'case1.OpenFOAM')

      # create a new 'Parallel UnstructuredGrid Writer'
      parallelUnstructuredGridWriter1 = servermanager.writers.XMLPUnstructuredGridWriter(Input=case1OpenFOAM)

      # register the writer with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the data, etc.
      coprocessor.RegisterWriter(parallelUnstructuredGridWriter1, filename='slice_%t.pvtu', freq=1, paddingamount=0)

      # ----------------------------------------------------------------
      # setup the visualization in view 'renderView1'
      # ----------------------------------------------------------------

      # show data from case1OpenFOAM
      case1OpenFOAMDisplay = Show(case1OpenFOAM, renderView1)

      # get color transfer function/color map for 'p'
      pLUT = GetColorTransferFunction('p')
      pLUT.RGBPoints = [-0.0006948409718461335, 0.231373, 0.298039, 0.752941, 3.34553187713027e-05, 0.865003, 0.865003, 0.865003, 0.0007617516093887389, 0.705882, 0.0156863, 0.14902]
      pLUT.ScalarRangeInitialized = 1.0

      # get opacity transfer function/opacity map for 'p'
      pPWF = GetOpacityTransferFunction('p')
      pPWF.Points = [-0.0006948409718461335, 0.0, 0.5, 0.0, 0.0007617516093887389, 1.0, 0.5, 0.0]
      pPWF.ScalarRangeInitialized = 1

      # trace defaults for the display properties.
      case1OpenFOAMDisplay.Representation = 'Surface'
      case1OpenFOAMDisplay.ColorArrayName = ['POINTS', 'p']
      case1OpenFOAMDisplay.LookupTable = pLUT
      case1OpenFOAMDisplay.OSPRayScaleArray = 'p'
      case1OpenFOAMDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
      case1OpenFOAMDisplay.SelectOrientationVectors = 'U'
      case1OpenFOAMDisplay.ScaleFactor = 75.0
      case1OpenFOAMDisplay.SelectScaleArray = 'p'
      case1OpenFOAMDisplay.GlyphType = 'Arrow'
      case1OpenFOAMDisplay.GlyphTableIndexArray = 'p'
      case1OpenFOAMDisplay.GaussianRadius = 3.75
      case1OpenFOAMDisplay.SetScaleArray = ['POINTS', 'p']
      case1OpenFOAMDisplay.ScaleTransferFunction = 'PiecewiseFunction'
      case1OpenFOAMDisplay.OpacityArray = ['POINTS', 'p']
      case1OpenFOAMDisplay.OpacityTransferFunction = 'PiecewiseFunction'
      case1OpenFOAMDisplay.DataAxesGrid = 'GridAxesRepresentation'
      case1OpenFOAMDisplay.SelectionCellLabelFontFile = ''
      case1OpenFOAMDisplay.SelectionPointLabelFontFile = ''
      case1OpenFOAMDisplay.PolarAxes = 'PolarAxesRepresentation'
      case1OpenFOAMDisplay.ScalarOpacityFunction = pPWF
      case1OpenFOAMDisplay.ScalarOpacityUnitDistance = 15.515969261125516

      # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
      case1OpenFOAMDisplay.DataAxesGrid.XTitleFontFile = ''
      case1OpenFOAMDisplay.DataAxesGrid.YTitleFontFile = ''
      case1OpenFOAMDisplay.DataAxesGrid.ZTitleFontFile = ''
      case1OpenFOAMDisplay.DataAxesGrid.XLabelFontFile = ''
      case1OpenFOAMDisplay.DataAxesGrid.YLabelFontFile = ''
      case1OpenFOAMDisplay.DataAxesGrid.ZLabelFontFile = ''

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      case1OpenFOAMDisplay.PolarAxes.PolarAxisTitleFontFile = ''
      case1OpenFOAMDisplay.PolarAxes.PolarAxisLabelFontFile = ''
      case1OpenFOAMDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
      case1OpenFOAMDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

      # setup the color legend parameters for each legend in this view

      # get color legend/bar for pLUT in view renderView1
      pLUTColorBar = GetScalarBar(pLUT, renderView1)
      pLUTColorBar.Title = 'p'
      pLUTColorBar.ComponentTitle = ''
      pLUTColorBar.TitleFontFile = ''
      pLUTColorBar.LabelFontFile = ''

      # set color bar visibility
      pLUTColorBar.Visibility = 1

      # show color legend
      case1OpenFOAMDisplay.SetScalarBarVisibility(renderView1, True)

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(parallelUnstructuredGridWriter1)
      # ----------------------------------------------------------------
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'case1.OpenFOAM': [1, 1]}
  coprocessor.SetUpdateFrequencies(freqs)
  if requestSpecificArrays:
    arrays = [['delC_0', 0], ['delC_1', 0], ['delC_1_0', 0], ['delC_2', 0], ['delC_2_0', 0], ['delC_3', 0], ['delC_3_0', 0], ['delC_4', 0], ['delC_4_0', 0], ['delC_5', 0], ['delC_5_0', 0], ['delC_6', 0], ['delC_6_0', 0], ['delPassive', 0], ['delSalinity', 0], ['delSalinity_0', 0], ['delT', 0], ['delT_0', 0], ['epsilon', 0], ['epsilon_0', 0], ['k', 0], ['k_0', 0], ['nut', 0], ['p', 0], ['rho', 0], ['U', 0], ['U_0', 0], ['delC_0', 1], ['delC_1', 1], ['delC_1_0', 1], ['delC_2', 1], ['delC_2_0', 1], ['delC_3', 1], ['delC_3_0', 1], ['delC_4', 1], ['delC_4_0', 1], ['delC_5', 1], ['delC_5_0', 1], ['delC_6', 1], ['delC_6_0', 1], ['delPassive', 1], ['delSalinity', 1], ['delSalinity_0', 1], ['delT', 1], ['delT_0', 1], ['epsilon', 1], ['epsilon_0', 1], ['k', 1], ['k_0', 1], ['nut', 1], ['p', 1], ['rho', 1], ['U', 1], ['U_0', 1]]
    coprocessor.SetRequestedArrays('case1.OpenFOAM', arrays)
  coprocessor.SetInitialOutputOptions(timeStepToStartOutputAt,forceOutputAtFirstCall)

  if rootDirectory:
      coprocessor.SetRootDirectory(rootDirectory)

  if make_cinema_table:
      coprocessor.EnableCinemaDTable()

  return coprocessor


#--------------------------------------------------------------
# Global variable that will hold the pipeline for each timestep
# Creating the CoProcessor object, doesn't actually create the ParaView pipeline.
# It will be automatically setup when coprocessor.UpdateProducers() is called the
# first time.
coprocessor = CreateCoProcessor()

#--------------------------------------------------------------
# Enable Live-Visualizaton with ParaView and the update frequency
coprocessor.EnableLiveVisualization(False, 1)

# ---------------------- Data Selection method ----------------------

def RequestDataDescription(datadescription):
    "Callback to populate the request for current timestep"
    global coprocessor

    # setup requests for all inputs based on the requirements of the
    # pipeline.
    coprocessor.LoadRequestedData(datadescription)

# ------------------------ Processing method ------------------------

def DoCoProcessing(datadescription):
    "Callback to do co-processing for current timestep"
    global coprocessor

    # Update the coprocessor by providing it the newly generated simulation data.
    # If the pipeline hasn't been setup yet, this will setup the pipeline.
    coprocessor.UpdateProducers(datadescription)

    # Write output data, if appropriate.
    coprocessor.WriteData(datadescription);

    # Write image capture (Last arg: rescale lookup table), if appropriate.
    coprocessor.WriteImages(datadescription, rescale_lookuptable=rescale_lookuptable,
        image_quality=0, padding_amount=imageFileNamePadding)

    # Live Visualization, if enabled.
    coprocessor.DoLiveVisualization(datadescription, "localhost", 22222)
