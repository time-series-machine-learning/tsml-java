/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package experiments;

import experiments.Experiments.ExperimentalArguments;
import experiments.data.DatasetLoading;
import java.io.File;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import tsml.filters.shapelet_filters.ShapeletFilter;
import tsml.transformers.Transformer;
import tsml.transformers.shapelet_tools.ShapeletTransformTimingUtilities;
import static tsml.transformers.shapelet_tools.ShapeletTransformTimingUtilities.nanoToOp;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchFactory;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

/**
 *
 * @author Aaron Bostrom - Travis Test
 */
public class TransformExperiments {
    
    private final static Logger LOGGER = Logger.getLogger(TransformExperiments.class.getName());
    
    public static boolean debug = false;

    public static void main(String[] args) throws Exception {
        System.out.println("Tony Dev Test");
        if (args.length > 0) {
            ExperimentalArguments expSettings = new ExperimentalArguments(args);
            SetupTransformExperiment(expSettings);
        }else{
            String[] settings=new String[7];
//Location of data set
            //settings[0]="-dp=E:/Data/TSCProblems2018/";//Where to get data 
            settings[0]="-dp=D:/Research TSC/Data/TSCProblems2018/";
            //settings[1]="-rp=E:/Results/";//Where to write results      
            settings[1]="-rp=D:/Research TSC/Results/";    
            settings[2]="-gtf=false"; //Whether to generate train files or not               
            settings[3]="-cn=ShapeletTransform"; //Classifier name
//                for(String str:DataSets.tscProblems78){
                settings[4]="-dn=SonyAIBORobotSurface2"; //Problem file   
                settings[5]="-f=2";//Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)  
                settings[6]= "-ctrh=1";
            System.out.println("Manually set args:");
            for (String str : settings)
                System.out.println("\t"+str);

            ExperimentalArguments expSettings = new ExperimentalArguments(settings);
            SetupTransformExperiment(expSettings);
//                }
            }
        }

    public static void SetupTransformExperiment(ExperimentalArguments expSettings) throws Exception {
        
        if (debug)
            LOGGER.setLevel(Level.FINEST);
        else 
            LOGGER.setLevel(Level.INFO);
        LOGGER.log(Level.FINE, expSettings.toString());

        long hrs = TimeUnit.HOURS.convert(expSettings.contractTrainTimeNanos, TimeUnit.NANOSECONDS);
        
        //Build/make the directory to write the train and/or testFold files to
        String partialWriteLocation = expSettings.resultsWriteLocation + expSettings.classifierName + hrs + "/";
        String transformWriteLocation = partialWriteLocation + "Transforms/" + expSettings.datasetName + "/";
        String additionalWriteLocation =  partialWriteLocation + /*expSettings.classifierName*/ "Shapelets" + "/" + expSettings.datasetName + "/";
        
        System.out.println(transformWriteLocation);
        File f = new File(transformWriteLocation);
        if (!f.exists())
            f.mkdirs();
                
        if (experiments.CollateResults.validateSingleFoldFile(transformWriteLocation) && experiments.CollateResults.validateSingleFoldFile(additionalWriteLocation)) {
            LOGGER.log(Level.INFO, expSettings.toShortString() + " already exists at "+additionalWriteLocation+", exiting.");
            LOGGER.log(Level.INFO, expSettings.toShortString() + " already exists at "+transformWriteLocation+", exiting.");
        }
        else{
            Transformer transformer = TransformLists.setTransform(expSettings);
            Instances[] data = DatasetLoading.sampleDataset(expSettings.dataReadLocation, expSettings.datasetName, expSettings.foldId);
             
            runExperiment(expSettings, data[0], data[1], transformer, transformWriteLocation, additionalWriteLocation);
            LOGGER.log(Level.INFO, "Experiment finished {0}", expSettings.toShortString());
         }
    }
    
    
    public static void runExperiment(ExperimentalArguments expSettings, Instances train, Instances test, Transformer transformer, String fullWriteLocation, String additionalDataFilePath) throws Exception{
        
            //this is hacky, but will do.
            Instances[] transforms = setContractDataAndProcess(expSettings, train, test, transformer);
        
            //Filter.useFilter is wekas weird way
            Instances transformed_train = transforms[0];
            Instances transformed_test = transforms[1];

            ArffSaver saver = new ArffSaver();

            String transformed_train_output = fullWriteLocation + expSettings.datasetName +"_TRAIN.arff";
            String transformed_test_output = fullWriteLocation + expSettings.datasetName +"_TEST.arff";

            saver.setInstances(transformed_train);
            saver.setFile(new File(transformed_train_output));
            saver.writeBatch();

            saver.setInstances(transformed_test);
            saver.setFile(new File(transformed_test_output));
            saver.writeBatch();

            
            writeAdditionalTransformData(expSettings, transformer, additionalDataFilePath);
    }
    
    
    private static Instances[] setContractDataAndProcess(ExperimentalArguments expSettings, Instances train, Instances test, Transformer transformer){
        
        Instances[] out = new Instances[2];
        
        switch(expSettings.classifierName){
            
            case"ST": case "ShapeletTransform":
                
                
                /*TODO: Can tidy it up big time. Or move some of this else where.*/
                ShapeletFilter st = (ShapeletFilter)transformer;
                //do contracting.
                int m = train.numAttributes()-1;
                int n = train.numInstances();
                ShapeletSearch.SearchType searchType = ShapeletSearch.SearchType.FULL;
                
                //kShapelets
                int numShapeletsInTransform=st.getNumberOfShapelets();
                
                long numShapeletsToSearchFor = 0;
                if(expSettings.contractTrainTimeNanos > 0){
                    long time  = expSettings.contractTrainTimeNanos; //time in nanoseconds for the number of hours we want to run for.
                    
                    //proportion of operations we can perform in time frame.
                    BigInteger opCountTarget = new BigInteger(Long.toString(time / nanoToOp));
                    BigInteger opCount = ShapeletTransformTimingUtilities.calculateOps(n, m, 1, 1);
                   
                    BigDecimal oct = new BigDecimal(opCountTarget);
                    BigDecimal oc = new BigDecimal(opCount);
                    BigDecimal prop = oct.divide(oc, MathContext.DECIMAL64);
                        
                    //proportion of shapelets vs. total no. shapelets.
                    numShapeletsToSearchFor = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n,m,3,m);
                    
                    //no point in searching more than full?
                    if(prop.doubleValue() < 1.0){
                        numShapeletsToSearchFor *= prop.doubleValue();
                        System.out.println(numShapeletsToSearchFor);
                        //make sure the k shapelets is less than the amount we're looking at.
                        numShapeletsInTransform =  numShapeletsToSearchFor > numShapeletsInTransform ? numShapeletsInTransform : (int) numShapeletsToSearchFor;
                        searchType = ShapeletSearch.SearchType.IMPROVED_RANDOM;
                    }
                }
                ShapeletSearchOptions sops = new ShapeletSearchOptions.Builder()
                        .setSearchType(searchType)
                        .setMin(3).setMax(m)
                        .setSeed(expSettings.foldId)
                        .setNumShapeletsToEvaluate(numShapeletsToSearchFor)
                        .build();

                st.setSearchFunction(new ShapeletSearchFactory(sops).getShapeletSearch());
                st.setNumberOfShapelets(numShapeletsInTransform);

                out[0] = st.process(train);
                out[1] = st.process(test);
                
                break;
            
           default:
                System.out.println("UNKNOWN CLASSIFIER "+transformer);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        
        
        return out;
    }

    private static void writeAdditionalTransformData(ExperimentalArguments expSettings, Transformer transformer, String additionalDataFilePath) {
                    
                    
        switch(expSettings.classifierName){
            
            case"ST": case "ShapeletTransform":
                ShapeletFilter st = (ShapeletFilter) transformer;
                st.writeAdditionalData(additionalDataFilePath, expSettings.foldId);
                break;
            
           default:
                System.out.println("UNKNOWN CLASSIFIER "+transformer);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
    }
    
    
}
