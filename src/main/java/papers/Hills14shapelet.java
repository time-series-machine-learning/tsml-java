/**
 * Code to recreate the results in the paper
 * @article{hills14shapelet,
  title={Classification of time series by shapelet transformation},
  author={J. Hills  and  J. Lines and E. Baranauskas and J. Mapp and A. Bagnall},
  journal={Data Mining and Knowledge Discovery},
  volume={28},
  issue={4},
  pages={851--881},
  year={2014}
}

 */

package papers;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransform;
import static timeseriesweka.filters.shapelet_transforms.quality_measures.ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IB1;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.shapelet_trees.*;

import weka.core.Instances;

public class Hills14shapelet {

    // There are two types of dataset assessment - LOOCV or Train/Test split
    private enum AssesmentType{LOOCV, TRAIN_TEST};
    
    // An array containing all the file names of the datasets used in the experiments
    private static String[] fileNames={	
                                    //Number of train, test cases, length, classes
        "Adiac",                    // 390,391,176,37
        "Beef",                     // 30,30,470,5
        "ChlorineConcentration",    // 467,3840,166,3
        "Coffee",                   // 28,28,286,2
        "DiatomSizeReduction",      // 16,306,345,4
        "DP_Little",                // 400,645,250,3  
        "DP_Middle",                // 400,645,250,3
        "DP_Thumb",                 // 400,645,250,3
        "ECGFiveDays",              // 23,861,136,2
        "ElectricDevices",          // 8953,7745,96,7
        "FaceFour",                 // 24,88,350,4
        "GunPoint",                 // 50,150,150,2
        "ItalyPowerDemand",         // 67,1029,24,2
        "Lighting7",                // 70,73,319,7
        "MedicalImages",            // 381,760,99,10
        "MoteStrain",               // 20,1252,84,2
        "MP_Little",                // 400,645,250,3 
        "MP_Middle",                // 400,645,250,3
        "PP_Little",                // 400,645,250,3
        "PP_Middle",                // 400,645,250,3
        "PP_Thumb",                 // 400,645,250,3
        "SonyAIBORobotSurface",     // 20,601,70,2
        "Symbols",                  // 25,995,398,6
        "SyntheticControl",         // 300,300,60,6
        "Trace",                    // 100,100,275,4
        "TwoLeadECG",               // 23,1139,82,2
        "Herrings",               // 64,64,512,2
//        "Herring500",             // 100,100,500,2
        "SyntheticData",            // 10,1000,500,2
 //       "Beetle-fly",               // 40,40,512,2
 //       "Bird-chicken",             // 40,40,512,2
 //       "ShapesAll"                 // 600,600,512,60 
    };
      private static final String userPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";  
    // Paths of the folders where the datasets are stored
    private static String[] filePaths={
        userPath+"Adiac/",                    // Adiac
        userPath+"Beef/",                     // Beef
        userPath+"ChlorineConcentration/",    // ChlorineConcentration
        userPath+"Coffee/",                   //Coffee
        userPath+"DiatomSizeReduction/",      // DiatomSizeReduction
        userPath+"Bones/",                    // "DP_Little"
        userPath+"Bones/",                    // "DP_Middle"
        userPath+"Bones/",                    // "DP_Thumb"
        userPath+"ECGFiveDays/",              // ECGFiveDays
        userPath+"ElectricDevices/",          // ElectricDevices
        userPath+"FaceFour/",                 // FaceFour
        userPath+"GunPoint/",                 // GunPoint
        userPath+"ItalyPowerDemand/",         // ItalyPowerDemand
        userPath+"Lighting7/",                // Lighting7
        userPath+"MedicalImages/",            // MedicalImages
        userPath+"MoteStrain/",               // MoteStrain
        userPath+"Bones/",                    // MP_Little
        userPath+"Bones/",                    // MP_Middle
        userPath+"Bones/",                    // PP_Little
        userPath+"Bones/",                    // PP_Middle
        userPath+"Bones/",                    // PP_Thumb
        userPath+"SonyAIBORobotSurface/",     // SonyAIBORobotSurface
        userPath+"Symbols/",                  // Symbols
        userPath+"SyntheticControl/",         // SyntheticControl
        userPath+"Trace/",                    // Trace
        userPath+"TwoLeadECG/",               // TwoLeadECG
        userPath+"Otoliths/",                 // Herrings
  //      userPath+"Otoliths/",                 // Herring500
        userPath+"SyntheticData/",            // SyntheticData
        userPath+"MPEG7Shapes/",              // Beetle-fly
        userPath+"MPEG7Shapes/",              // Bird-chicken
   //     userPath+"MPEG7Shapes/"               // ShapesAll
    };

    // An array containing the assesment type for each of the datasets. 
    private static AssesmentType[] assesmentTypes = {
        AssesmentType.TRAIN_TEST,   // Adiac
        AssesmentType.TRAIN_TEST,   // Beef
        AssesmentType.TRAIN_TEST,   // ChlorineConcentration
        AssesmentType.TRAIN_TEST,   // Coffee
        AssesmentType.TRAIN_TEST,   // DiatomSizeReduction
        AssesmentType.TRAIN_TEST,   // DP_Little 
        AssesmentType.TRAIN_TEST,   // DP_Middle
        AssesmentType.TRAIN_TEST,   // DP_Thumb
        AssesmentType.TRAIN_TEST,   // ECGFiveDays
        AssesmentType.TRAIN_TEST,   // ElectricDevices
        AssesmentType.TRAIN_TEST,   // FaceFour
        AssesmentType.TRAIN_TEST,   // GunPoint
        AssesmentType.TRAIN_TEST,   // ItalyPowerDemand
        AssesmentType.TRAIN_TEST,   // Lighting7
        AssesmentType.TRAIN_TEST,   // MedicalImages
        AssesmentType.TRAIN_TEST,   // MoteStrain
        AssesmentType.TRAIN_TEST,   // MP_Little
        AssesmentType.TRAIN_TEST,   // MP_Middle
        AssesmentType.TRAIN_TEST,   // PP_Little
        AssesmentType.TRAIN_TEST,   // PP_Middle
        AssesmentType.TRAIN_TEST,   // PP_Thumb
        AssesmentType.TRAIN_TEST,   // SonyAIBORobotSurface
        AssesmentType.TRAIN_TEST,   // Symbols
        AssesmentType.TRAIN_TEST,   // SyntheticControl
        AssesmentType.TRAIN_TEST,   // Trace
        AssesmentType.TRAIN_TEST,   // TwoLeadECG
        AssesmentType.TRAIN_TEST,   // Herrings
        AssesmentType.LOOCV,        // Herring500
        AssesmentType.TRAIN_TEST,   // SyntheticData
        AssesmentType.LOOCV,        // Beetle-fly
        AssesmentType.LOOCV,        // Bird-chicken
        AssesmentType.TRAIN_TEST    // ShapesAll
    };
    
    // An array containing the shapelet min-max interval for each of the datasets. 
    private static int[][] shapeletMinMax = {
        {3, 10},    // Adiac
        {8, 30},    // Beef
        {7, 20},    // ChlorineConcentration
        {18,30},    // Coffee
        {7,16},     // DiatomSizeReduction
        {9, 36},    // DP_Little 
        {15, 43},   // DP_Middle
        {11, 47},   // DP_Thumb
        {24, 76},   // ECGFiveDays
        {10, 42},   // ElectricDevices
        {20, 120},  // FaceFour
        {24, 55},   // GunPoint
        {7, 14},    // ItalyPowerDemand
        {20, 80},   // Lighting7
        {9, 35},    // MedicalImages
        {16, 31},   // MoteStrain
        {15, 41},   // MP_Little
        {20, 53},   // MP_Middle
        {13, 38},   // PP_Little
        {14, 34},   // PP_Middle
        {14, 41},   // PP_Thumb
        {15, 36},   // SonyAIBORobotSurface
        {52, 155},  // Symbols
        {20, 56},   // SyntheticControl
        {62, 232},  // Trace
        {7, 13},    // TwoLeadECG
        {30, 101},  // Herrings
        {30, 101},  // Herring500
        {25, 35},   // SyntheticData
        {30, 101},  // Beetle-fly
        {30, 101},  // Bird-chicken
        {30, 110}   // ShapesAll
    };
    
    // Variables for holding filters for data transformation
    private static ShapeletTransform shapeletFilter;
    
    // Variables for holding data
    private static Instances[] instancesTrain;
    private static Instances[] instancesTest;
    
    // Variables for holding the classifier information
    private static Classifier classifiers[];
    private static String classifierNames[];
    
    // Variables for holding user input
    private static int tableToProduceIndex;
    private static String outFileName;
    private static int fileToProcessIndex;
    private static int classifierToProcessIndex;
    
    // Method to load the datasets.
    private static void loadData(){
        instancesTrain = new Instances[fileNames.length];
        instancesTest = new Instances[fileNames.length];
            
        //Load all the datasets and set class index for loaded instances
        for(int i=0; i<fileNames.length; i++){
            
            // Load test/train splits
            if(assesmentTypes[i] == AssesmentType.TRAIN_TEST){
                instancesTrain[i] = ShapeletTransform.loadData(filePaths[i]+fileNames[i]+"_TRAIN.arff");
                instancesTest[i] = ShapeletTransform.loadData(filePaths[i]+fileNames[i]+"_TEST.arff");
            }else if(assesmentTypes[i] == AssesmentType.LOOCV){
                instancesTrain[i] = ShapeletTransform.loadData(filePaths[i]+fileNames[i]+".arff");
                instancesTest[i] = null;   
            }
            
            // Set class indices
            instancesTrain[i].setClassIndex(instancesTrain[i].numAttributes() - 1);
            if(assesmentTypes[i] == AssesmentType.TRAIN_TEST){
                instancesTest[i].setClassIndex(instancesTest[i].numAttributes() - 1);
            }
        }     
    }
    
    public static void table2() throws Exception{
   
        // Initialise classifiers required for this experiment
        classifiers = new Classifier[4];
        classifiers[0] = new ShapeletTreeClassifier("infoTree.txt");
        classifiers[1] = new KruskalWallisTree("kwTree.txt");
        classifiers[2] = new MoodsMedianTreeWithInfoGain("mmWithInfoTree.txt");
        classifiers[3] = new FStatShapeletTreeWithInfoGain("fStatTree.txt");
        
        // Set up names for the classifiers - only used for output
        classifierNames = new String[4];
        classifierNames[0] = "IG";
        classifierNames[1] = "KruskalWallis";
        classifierNames[2] = "MoodMedIG";
        classifierNames[3] = "F-stat";
              
        if((classifierToProcessIndex < 1 || classifierToProcessIndex > classifiers.length) && classifierToProcessIndex != -1 ){
            throw new IOException("Invalid classifier identifier.");
        }else{
           if(classifierToProcessIndex != -1){
                classifierToProcessIndex--; 
           }
        }
        
        // Compute classifier accuracies for each classifier
        double accuracies[][] = new double[classifiers.length][];
        for(int i = 0; i < classifiers.length; i++){
            if(i == classifierToProcessIndex || classifierToProcessIndex == -1){
                accuracies[i] = classifierAccuracy(i, false, true, false);
            }
        }
        
        // Write experiment output to file 
        writeFileContent(accuracies);
    }
   
    public static void table3() throws Exception{
        
        // Initialise classifiers required for this experiment
        classifiers = new Classifier[4];
        classifiers[0] = new ShapeletTreeClassifier("infoTree.txt");
        classifiers[1] = new FStatShapeletTreeWithInfoGain("fStatTree.txt");
        classifiers[2] = new KruskalWallisTree("kwTree.txt");
        classifiers[3] = new MoodsMedianTree("mmTree.txt");
        
        // Set up names for the classifiers - only used for output
        classifierNames = new String[4];
        classifierNames[0] = "Information Gain";
        classifierNames[1] = "F-stat";
        classifierNames[2] = "Kruskal-Wallis";
        classifierNames[3] = "Mood's Median";
        
        if((classifierToProcessIndex < 1 || classifierToProcessIndex > classifiers.length) && classifierToProcessIndex != -1 ){
            throw new IOException("Invalid classifier identifier.");
        }else{
           if(classifierToProcessIndex != -1){
                classifierToProcessIndex--; 
           }
        }
                              
        // Record classifier times to find single shapelet
        double times[][] = new double[classifiers.length][instancesTrain.length];
        for(int i = 0; i < classifiers.length; i++){
            if(i == classifierToProcessIndex || classifierToProcessIndex == -1){
                for(int j = 0; j < instancesTrain.length; j++){
                    if(fileToProcessIndex == j || fileToProcessIndex == -1){
                        // Get training data
                        Instances data = null;
                        if(assesmentTypes[j] == AssesmentType.TRAIN_TEST){
                            data = instancesTrain[j];
                        }else if(assesmentTypes[j] == AssesmentType.LOOCV){
                            data =  instancesTrain[j].trainCV(instancesTrain[j].numInstances(), 0);
                        }

                        // Store time
                        if(classifiers[i] instanceof ShapeletTreeClassifier){
                            times[i][j] = ((ShapeletTreeClassifier)classifiers[i]).timingForSingleShapelet(data, shapeletMinMax[j][0], shapeletMinMax[j][1]);
                        }else if(classifiers[i] instanceof KruskalWallisTree){
                            times[i][j] = ((KruskalWallisTree)classifiers[i]).timingForSingleShapelet(data, shapeletMinMax[j][0], shapeletMinMax[j][1]);   
                        }else if(classifiers[i] instanceof MoodsMedianTree){
                            times[i][j] = ((MoodsMedianTree)classifiers[i]).timingForSingleShapelet(data, shapeletMinMax[j][0], shapeletMinMax[j][1]);
                        }else if(classifiers[i] instanceof MoodsMedianTreeWithInfoGain){
                            times[i][j] = ((MoodsMedianTreeWithInfoGain)classifiers[i]).timingForSingleShapelet(data, shapeletMinMax[j][0], shapeletMinMax[j][1]); 
                        }else if(classifiers[i] instanceof FStatShapeletTreeWithInfoGain){
                            times[i][j] = ((FStatShapeletTreeWithInfoGain)classifiers[i]).timingForSingleShapelet(data, shapeletMinMax[j][0], shapeletMinMax[j][1]);
                        }
                    }
                }
            }
        }
        
        // Write experiment output to file 
        writeFileContent(times);
    }
    
    public static void table4_5() throws Exception{
        
        // Initialise classifiers required for this experiment
        classifiers = new Classifier[8];
        classifiers[0] = new ShapeletTreeClassifier("infoTree.txt");
        classifiers[1] = new J48();
        classifiers[2] = new IB1();
        classifiers[3] = new NaiveBayes();
        classifiers[4] = new BayesNet();
        classifiers[5] = new RandomForest();
        classifiers[6] = new RotationForest();
        classifiers[7] = new SMO();

        // Set up names for the classifiers - only used for output
        classifierNames = new String[8];
        classifierNames[0] = "ShapeletTree";
        classifierNames[1] = "C4.5";
        classifierNames[2] = "1NN";
        classifierNames[3] = "Naive Bayes";
        classifierNames[4] = "Bayesian Network";
        classifierNames[5] = "Random Forest";
        classifierNames[6] = "Rotation Forest";
        classifierNames[7] = "SVM (linear)";

        if((classifierToProcessIndex < 1 || classifierToProcessIndex > classifiers.length) && classifierToProcessIndex != -1 ){
            throw new IOException("Invalid classifier identifier.");
        }else{
           if(classifierToProcessIndex != -1){
                classifierToProcessIndex--; 
           }
        }
           
        // Compute classifier accuracies for each classifier
        double accuracies[][] = new double[classifiers.length][];
        boolean transFlag = false;
        for(int i = 0; i < classifiers.length; i++){
            if(!(classifiers[i] instanceof ShapeletTreeClassifier)){
                //shapeletFilter = new Shapelet();
                //shapeletFilter.setQualityMeasure(Shapelet.ShapeletQualityChoice.INFORMATION_GAIN);
                //shapeletFilter.supressOutput();
                transFlag = true;
            }
            if(i == classifierToProcessIndex || classifierToProcessIndex == -1){
                accuracies[i] = classifierAccuracy(i, transFlag, false, true);
            }
        }
        
        // Write experiment output to file 
        writeFileContent(accuracies);
    }
        
    /**
     * A method to validate a given classifier
     * 
     * @param classifierIndex index of the classifier to be validated
     * @param useTransformedData flag indicating what type of data to use. 
     *                           Shapelet is used for data transformation.
     * @param computeErrorRate flag indicating whether error rate is required
     *                         rather than classifier accuracy.
     * @param usePercentage flag indicating whether an accuracy/error rate should
     *                      be converted to percentage.
     * @return classifier accuracy/error rate
     */
    private static double[] classifierAccuracy(int classifierIndex, boolean useTransformedData, boolean computeErrorRate, boolean usePercentage){

        // Array for storing the classifier accuracies
        double[] accuracies = new double[instancesTrain.length];
        
        // Generate average accuracies
        for (int n = 0; n < instancesTrain.length; n++){
            if(fileToProcessIndex == n || fileToProcessIndex == -1){
                try{
                    if(assesmentTypes[n] == AssesmentType.TRAIN_TEST){
                        accuracies[n] = classiferAccuracyTrainTest(classifierIndex, n, useTransformedData);
                    }else if(assesmentTypes[n] == AssesmentType.LOOCV){
                        accuracies[n] = classiferAccuracyLOOCV(classifierIndex, n, useTransformedData);
                    }
                }catch(Exception e){
                    e.printStackTrace();
                }
                
                if(computeErrorRate){
                    accuracies[n] = 1 - accuracies[n];
                }
                
                if(usePercentage){
                    accuracies[n] *= 100;
                }
            }
        }
             
        return accuracies;
    }
        
    /**
     * A method to perform simple train/test split validation using given classifier
     * and data.
     * 
     * @param classifierIndex index of the classifier to be used in validation.
     * @param dataIndex index of the data to be used in validation.
     * @param trainData data to be used to build the classifier.
     * @param testData data to be used to test the classifier
     * @return accuracy of the classifier.
     */
    private static double classiferAccuracyTrainTest(int classifierIndex, int dataIndex, boolean useTransformedData){

        double accuracy = 0.0;
        
        Instances trainData = null, testData = null;
        
        if(useTransformedData){
            //Initialize filter
            try{
                shapeletFilter = ShapeletTransform.createFilterFromFile(filePaths[dataIndex]+fileNames[dataIndex]+"_TRAIN_TRANS.txt", instancesTrain[dataIndex].numAttributes()/2);
                shapeletFilter.supressOutput();
            }catch (Exception e){
                shapeletFilter = new ShapeletTransform();
                shapeletFilter.setQualityMeasure(INFORMATION_GAIN);
                shapeletFilter.supressOutput();
                shapeletFilter.setNumberOfShapelets(instancesTrain[dataIndex].numAttributes()/2);
                shapeletFilter.setShapeletMinAndMax(shapeletMinMax[dataIndex][0], shapeletMinMax[dataIndex][1]);
                shapeletFilter.setLogOutputFile(filePaths[dataIndex]+fileNames[dataIndex]+"_TRAIN_TRANS.txt");
            }
            
            //Transform data
            try{
                trainData = shapeletFilter.process(instancesTrain[dataIndex]);
                testData = shapeletFilter.process(instancesTest[dataIndex]);
            }catch(Exception e){
                e.printStackTrace();
            }
        }else{
            trainData = instancesTrain[dataIndex];
            testData = instancesTest[dataIndex];
        }   
                
        // Build classifer using train split
        buildClassifier(classifierIndex, trainData, dataIndex);

        //Classify test instancs while recording accuracy
        for(int j = 0; j < testData.numInstances(); j++){

            double classifierPrediction = 0.0;
            try{
                classifierPrediction = classifiers[classifierIndex].classifyInstance(testData.instance(j));
            }catch(Exception e){
                e.printStackTrace();
            }
            
            double actualClass = testData.instance(j).classValue();

            if(classifierPrediction == actualClass) {
                accuracy++;
            }

            // Compute average accuracy if it is the last test instance
            if(j == testData.numInstances() - 1){
                accuracy /= testData.numInstances();
            }
        }
        
        return accuracy;
    }

    /**
     * A method to perform leave one out cross validation using given classifier and
     * data.
     * 
     * @param classifierIndex index of the classifier to be used in cross validation.
     * @param dataIndex index of the data to be used in cross validation.
     * @param data data to be used in cross validation.
     * @return accuracy of the classifier.
     */
    private static double classiferAccuracyLOOCV(int classifierIndex, int dataIndex, boolean useTransformedData){
               
        //Variables for holding folds
        Instances data = instancesTrain[dataIndex];
        Instances trainFold;
        Instances testFold;
        
        double accuracy = 0.0;
        
        //Generate average accuracies
        for (int n = 0; n < data.numInstances(); n++) {
            System.out.println("\n\n\n\n\nProcessing fold: " + n+"\n\n\n\n\n");
            //Generate folds
            trainFold = data.trainCV(data.numInstances(), n);
            testFold = data.testCV(data.numInstances(), n);
  
            if(useTransformedData){
                //Initialize filter
                try{
                    shapeletFilter = ShapeletTransform.createFilterFromFile(filePaths[dataIndex]+fileNames[dataIndex]+"_TRANS_"+n+".txt", instancesTrain[dataIndex].numAttributes()/2);
                    shapeletFilter.supressOutput();
                }catch (Exception e){
                    shapeletFilter = new ShapeletTransform();
                    shapeletFilter.setQualityMeasure(INFORMATION_GAIN);
                    shapeletFilter.supressOutput();
                    shapeletFilter.setNumberOfShapelets(instancesTrain[dataIndex].numAttributes()/2);
                    shapeletFilter.setShapeletMinAndMax(shapeletMinMax[dataIndex][0], shapeletMinMax[dataIndex][1]);
                    shapeletFilter.setLogOutputFile(filePaths[dataIndex]+fileNames[dataIndex]+"_TRANS_"+n+".txt");
                }

                //Transform data
                try{
                    trainFold = shapeletFilter.process(trainFold);
                    testFold = shapeletFilter.process(testFold);
                }catch(Exception e){
                    e.printStackTrace();
                }
            }
            
            // Build classifer using train fold
            buildClassifier(classifierIndex, trainFold, dataIndex);

            double classifierPrediction = 0.0;
            try{
                classifierPrediction = classifiers[classifierIndex].classifyInstance(testFold.instance(0));
            }catch(Exception e){
                e.printStackTrace();
            }
            
            double actualClass = testFold.instance(0).classValue();

            if(classifierPrediction == actualClass) {
                accuracy++;
            }
            
            // Compute average accuracy if it is the last test instance
            if(n == data.numInstances() - 1){
                accuracy /= data.numInstances();
            }
        }

        return accuracy;
    }
    
    /**
     * A method to build a classifier with given data.
     * 
     * @param classifierID classifier ID, which determines which classifier is built.
     * @param instances data to be used for building the classifier
     * @param dataSetIndex data set index, which determines what parameters are used
     *                     for building the classifier.
     */
    private static void buildClassifier(int classifierIndex, Instances instances, int dataSetIndex){
        // Set the shapelet min/max if the current classifer is a ShapeletTree or its variation
        if(classifiers[classifierIndex] instanceof ShapeletTreeClassifier){
            ((ShapeletTreeClassifier)classifiers[classifierIndex]).setShapeletMinMaxLength(shapeletMinMax[dataSetIndex][0], shapeletMinMax[dataSetIndex][1]);
        }else if(classifiers[classifierIndex] instanceof KruskalWallisTree){
            ((KruskalWallisTree)classifiers[classifierIndex]).setShapeletMinMaxLength(shapeletMinMax[dataSetIndex][0], shapeletMinMax[dataSetIndex][1]);   
        }else if(classifiers[classifierIndex] instanceof MoodsMedianTree){
            ((MoodsMedianTree)classifiers[classifierIndex]).setShapeletMinMaxLength(shapeletMinMax[dataSetIndex][0], shapeletMinMax[dataSetIndex][1]);
        }else if(classifiers[classifierIndex] instanceof MoodsMedianTreeWithInfoGain){
            ((MoodsMedianTreeWithInfoGain)classifiers[classifierIndex]).setShapeletMinMaxLength(shapeletMinMax[dataSetIndex][0], shapeletMinMax[dataSetIndex][1]); 
        }else if(classifiers[classifierIndex] instanceof FStatShapeletTreeWithInfoGain){
            ((FStatShapeletTreeWithInfoGain)classifiers[classifierIndex]).setShapeletMinMaxLength(shapeletMinMax[dataSetIndex][0], shapeletMinMax[dataSetIndex][1]); 
        }

        //Build classifier
        try{
            classifiers[classifierIndex].buildClassifier(instances); 
        }catch(Exception e){
            e.printStackTrace();
        }
    }
   
    /**
     * A method to write content to a given file.
     * 
     * @param fileName file name including extension
     * @param content  content of the file
     */
   private static void writeFileContent(double content[][]){
        // Check if file name is provided.
        if(outFileName == null || outFileName.isEmpty()){
            outFileName = "Table_" + tableToProduceIndex + 
                          "_File_" + (fileToProcessIndex+1) + 
                          "_Classifier_" + (classifierToProcessIndex+1) + ".csv";            
        }
            
       // If a file with given name does not exists then create one and print
       // the header to it, which inlcudes all the classifier names used in the
       // experiment. 
       StringBuilder sb = new StringBuilder();
        if(!isFileExists(outFileName)){
            sb.append("Data Set, "); 
            for(int i = 0; i < classifierNames.length; i++){
                if(i == classifierToProcessIndex || classifierToProcessIndex == -1){
                    sb.append(classifierNames[i]);
                }
                
                if(-1 == classifierToProcessIndex && i != classifierNames.length - 1){
                    sb.append(", ");
                }
            }
            writeToFile(outFileName, sb.toString(), false);
        }
        
        // Print the experiment results to the file.
        sb = new StringBuilder();
        for(int i = 0; i < fileNames.length; i++){
            if(fileToProcessIndex == i || fileToProcessIndex == -1){
                for(int k = 0; k < classifiers.length; k++){
                    if(k == 0){
                        sb.append(fileNames[i]);
                        sb.append(", ");
                    }
                    
                    if(k == classifierToProcessIndex || classifierToProcessIndex == -1 ){
                        sb.append(content[k][i]);
                    }
                    
                    if(-1 == classifierToProcessIndex && k != classifiers.length - 1){
                        sb.append(", ");
                    }
                }
            }
        }
        writeToFile(outFileName, sb.toString(), true);
   }
   
   /**
    * A method to write text into a file.
    * @param filename file name including the extension.
    * @param text content to be written into the file.
    * @param append flag indicating whether a file should be appended (true) or
    *        replaced (false).
    */
    private static void writeToFile(String filename, String text, boolean append) {
        
        BufferedWriter bufferedWriter = null;
        
        try {       
            //Construct the BufferedWriter object
            bufferedWriter = new BufferedWriter(new FileWriter(filename, append));
            
            //Start writing to the output stream
            bufferedWriter.write(text);
            bufferedWriter.newLine();
            
        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            //Close the BufferedWriter
            try {
                if (bufferedWriter != null) {
                    bufferedWriter.flush();
                    bufferedWriter.close();
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }
    
    /**
     * A method to check if file with a given name exists.
     * 
     * @param filename file name including the extension.
     * @return true if file with given file name exists, otherwise false.
     */
    private static boolean isFileExists(String filename){
        File f = new File(filename);
        if(f.isFile() && f.canWrite()) {
            return true;
        }else{
            return false;
        }   
    }
       
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception{
        
        tableToProduceIndex = 2;
        // Process user input
        try{
            tableToProduceIndex = Integer.parseInt(args[0]);
            outFileName = args[1];
            fileToProcessIndex = Integer.parseInt(args[2]);
            classifierToProcessIndex = Integer.parseInt(args[3]);
               
            // Check if file index is correct
            if(fileToProcessIndex < 1 || fileToProcessIndex > fileNames.length){
                throw new IOException("Invalid file identifier.");
            }else{
                fileToProcessIndex--;   // indexed from 1 when using arguments
            }
            
            
        }catch(Exception e){
            System.err.println("Invalid user input. Using default values");
            tableToProduceIndex = 2;        // refer to paper for indices    
            fileToProcessIndex = 0;         // indexed from 0 if setting here.
            classifierToProcessIndex = -1;  // -1 all classifiers
        }
                
        loadData();
 
        try{
            switch (tableToProduceIndex){
                case 2: table2(); break;
                case 3: table3(); break;
                case 4: table4_5(); break;
                case 5: table4_5(); break;
                default: throw new IOException("Unknow table identifier.");
            }
        }catch(Exception e){
            e.printStackTrace();
        }
    }
    
}
