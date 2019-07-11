
#--------------------------------------------------------------

# Global timestep output options
timeStepToStartOutputAt=0
forceOutputAtFirstCall=False

# Global screenshot output options
imageFileNamePadding=0
rescale_lookuptable=False

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
      # setup the data processing pipelines
      # ----------------------------------------------------------------

      # trace generated using paraview version 5.6.0
      #
      # To ensure correct image size when batch processing, please search 
      # for and uncomment the line `# renderView*.ViewSize = [*,*]`

      #### disable automatic camera reset on 'Show'
      paraview.simple._DisableFirstRenderCameraReset()

      # create a new 'OpenFOAMReader'
      # create a producer from a simulation input
      rNtestfoam = coprocessor.CreateProducer(datadescription, 'RNtest.foam')

      # create a new 'Parallel MultiBlockDataSet Writer'
      parallelMultiBlockDataSetWriter1 = servermanager.writers.XMLMultiBlockDataWriter(Input=rNtestfoam)

      # register the writer with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the data, etc.
      coprocessor.RegisterWriter(parallelMultiBlockDataSetWriter1, filename='filename_%t.vtm', freq=1, paddingamount=0)

      # create a new 'Threshold'
      threshold1 = Threshold(Input=rNtestfoam)
      threshold1.Scalars = ['CELLS', 'delC0']
      threshold1.ThresholdRange = [0.01, 100000.0]

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(threshold1)
      # ----------------------------------------------------------------
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'RNtest.foam': [1, 1]}
  coprocessor.SetUpdateFrequencies(freqs)
  if requestSpecificArrays:
    arrays = [['delC0', 0], ['delC0_0', 0], ['delC_1', 0], ['delC_1_0', 0], ['delC_2', 0], ['delC_2_0', 0], ['delC_3', 0], ['delC_3_0', 0], ['delC_4', 0], ['delC_4_0', 0], ['delC_5', 0], ['delC_5_0', 0], ['delC_6', 0], ['delC_6_0', 0], ['delPassive', 0], ['delSalinity', 0], ['delSalinity_0', 0], ['delT', 0], ['delT_0', 0], ['epsilon', 0], ['epsilon_0', 0], ['k', 0], ['k_0', 0], ['nut', 0], ['p', 0], ['rho', 0], ['U', 0], ['U_0', 0], ['delC0', 1], ['delC0_0', 1], ['delC_1', 1], ['delC_1_0', 1], ['delC_2', 1], ['delC_2_0', 1], ['delC_3', 1], ['delC_3_0', 1], ['delC_4', 1], ['delC_4_0', 1], ['delC_5', 1], ['delC_5_0', 1], ['delC_6', 1], ['delC_6_0', 1], ['delPassive', 1], ['delSalinity', 1], ['delSalinity_0', 1], ['delT', 1], ['delT_0', 1], ['epsilon', 1], ['epsilon_0', 1], ['k', 1], ['k_0', 1], ['nut', 1], ['p', 1], ['rho', 1], ['U', 1], ['U_0', 1]]
    coprocessor.SetRequestedArrays('RNtest.foam', arrays)
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
coprocessor.EnableLiveVisualization(True, 1)

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
