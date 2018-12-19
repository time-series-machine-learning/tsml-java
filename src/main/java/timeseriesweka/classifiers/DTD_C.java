
package timeseriesweka.classifiers;

import java.text.DecimalFormat;
import utilities.ClassifierTools;
import utilities.SaveParameterInfo;
import weka.classifiers.lazy.kNN;
import utilities.ClassifierResults;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import timeseriesweka.elastic_distance_measures.DTW_DistanceBasic;
import weka.filters.SimpleBatchFilter;
import timeseriesweka.filters.Cosine;
import timeseriesweka.filters.Sine;
import timeseriesweka.filters.Hilbert;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 * 
 * Implementation of:
 * Górecki, Tomasz, and Maciej Łuczak. 
 * Non-isometric transforms in time series classification using DTW
 * Knowledge-Based Systems 61 (2014): 98-108.
 * 
 */

/*
=========================================================================================================================
BRIEF DESCRIPTION:
=========================================================================================================================

The classifier is highly related to DD_DTW; however, instead of using a weighted combination of the raw 
data representation and derivatives, this classifier replaces derivatives with either Sine, Cosine or Hilbert-transformed
data. It should also be noted that unlike the aforementioned class, this classifier does not use ED, and uses only a
full-window DTW implementation. Two params are again used to weight the classifier, a and b, which represent the weight 
of standard DTW and transformed DTW respectively (note: only one transform is ever used at a time, so it is only DTW and 
cosDTW/sinDTW/hilDTW). The transformed-DTW is simply just the DTW distance measure being used to compute distances with
transformed data. 

As with the previous derivative iteration of this classifier, the params a and b are in the range of 0-1, and the sum of 
a+b is always 1 (i.e. binary split in the weighting between the two classifiers). Therefore a = 1, b = 0 is equivilent to 
just using DTW, and a=0, b=1 is equivilent DTW on the appropriately-transformed data. 

Again, the author's propose using a single parameter alpha to weight these two components by using it to derive values of 
a and b. This was ignored in this classifier however, as results indicated that this did not reproduce results for the 
derivative version of the classifier (see notes in DD_DTW.java). Therefore in our experiments we search 
from a = 0 to 1 and b = 1 to 0 in increments of 0.01 (101 param options) again.

=========================================================================================================================
HOW TO USE:
=========================================================================================================================
The class extends the kNN class, so classifier functionality is included. Three additional parameters should be set: 
1. the type of transform to use in the classifier (Cosine/Sine/Hilbert) (default is Cosine unless specified in constructor) 
2. values for a and b (defaults to a=1 and b=0, equiv to just DTW) 

The params a and b can be set explicitly through a mutator. However, if not specified, the buildClassifier method
performs the LOOCV procedure outlined in the original paper to find the values of a and b using the training data.

=========================================================================================================================
RECREATING RESULTS:
=========================================================================================================================
See method recreateResultsTable()
String DATA_DIR should be changed to point to the dir of TSC problems (examples included in code)

=========================================================================================================================
RELATED CLASSES:
=========================================================================================================================
Previous iteration:
DD_DTW.java 

Classes used:
Cosine.java, Sine.java, Hilbert.java


*/
public class DTD_C extends DD_DTW{

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "T. Górecki and M. Łuczak");
        result.setValue(TechnicalInformation.Field.TITLE, "Non-isometric transforms in time series classification using DTW");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Knowledge-Based Systems");
        result.setValue(TechnicalInformation.Field.VOLUME, "61");
        result.setValue(TechnicalInformation.Field.PAGES, "98-108");
        result.setValue(TechnicalInformation.Field.YEAR, "2014");
        return result;
    }    
    
    public static final String DATA_DIR = "C:/Temp/Dropbox/TSC Problems/";
//    public static final String DATA_DIR = "/Users/Jay/Dropbox/TSC Problems/";
    
    public static final double[] ALPHAS = {
        //<editor-fold defaultstate="collapsed" desc="alpha values">
        1,
        1.01,
        1.02,
        1.03,
        1.04,
        1.05,
        1.06,
        1.07,
        1.08,
        1.09,
        1.1,
        1.11,
        1.12,
        1.13,
        1.14,
        1.15,
        1.16,
        1.17,
        1.18,
        1.19,
        1.2,
        1.21,
        1.22,
        1.23,
        1.24,
        1.25,
        1.26,
        1.27,
        1.28,
        1.29,
        1.3,
        1.31,
        1.32,
        1.33,
        1.34,
        1.35,
        1.36,
        1.37,
        1.38,
        1.39,
        1.4,
        1.41,
        1.42,
        1.43,
        1.44,
        1.45,
        1.46,
        1.47,
        1.48,
        1.49,
        1.5,
        1.51,
        1.52,
        1.53,
        1.54,
        1.55,
        1.56,
        1.57
//</editor-fold>
    };
    
    public static final String[] PAPER_DATASETS = {
        //<editor-fold defaultstate="collapsed" desc="Datasets from the paper">
        "fiftywords", // 450,455,270,50
        "Adiac", // 390,391,176,37
        "Beef", // 30,30,470,5
        "Car", // 60,60,577,4
        "CBF", // 30,900,128,3
        "ChlorineConcentration", // 467,3840,166,3
        "CinC_ECG_torso", // 40,1380,1639,4
        "Coffee", // 28,28,286,2
        "Cricket_X", // 390,390,300,12
        "Cricket_Y", // 390,390,300,12
        "Cricket_Z", // 390,390,300,12
        "DiatomSizeReduction", // 16,306,345,4
        "ECGFiveDays", // 23,861,136,2
        "FaceAll", // 560,1690,131,14
        "FaceFour", // 24,88,350,4
        "FacesUCR", // 200,2050,131,14
        "fish", // 175,175,463,7
        "GunPoint", // 50,150,150,2
        "Haptics", // 155,308,1092,5
        "InlineSkate", // 100,550,1882,7
        "ItalyPowerDemand", // 67,1029,24,2
        "Lightning2", // 60,61,637,2
        "Lightning7", // 70,73,319,7
        "MALLAT", // 55,2345,1024,8
        "MedicalImages", // 381,760,99,10
        "MoteStrain", // 20,1252,84,2
        "NonInvasiveFatalECG_Thorax1", // 1800,1965,750,42
        "NonInvasiveFatalECG_Thorax2", // 1800,1965,750,42
        "OliveOil", // 30,30,570,4
        "OSULeaf", // 200,242,427,6
        "Plane", // 105,105,144,7
        "SonyAIBORobotSurface", // 20,601,70,2
        "SonyAIBORobotSurfaceII", // 27,953,65,2
        "StarLightCurves", // 1000,8236,1024,3
        "SwedishLeaf", // 500,625,128,15
        "Symbols", // 25,995,398,6
        "SyntheticControl", // 300,300,60,6
        "Trace", // 100,100,275,4
        "TwoPatterns", // 1000,4000,128,4
        "TwoLeadECG", // 23,1139,82,2
        "UWaveGestureLibrary_X", // 896,3582,315,8
        "UWaveGestureLibrary_Y", // 896,3582,315,8
        "UWaveGestureLibrary_Z", // 896,3582,315,8
        "wafer", // 1000,6164,152,2
        "WordSynonyms", // 267,638,270,25
        "yoga" // 300,3000,426,2
        //</editor-fold>
    };
    
    public static enum TransformType{SIN,COS,HIL};
      
    private TransformType transformType;
    
    
    public DTD_C(){
        super();
        this.transformType = TransformType.COS;
        this.distanceFunction = new TransformWeightedDTW(this.transformType);
    }
    
    public DTD_C(TransformType transformType){
        super();
        this.transformType = transformType;
        this.distanceFunction = new TransformWeightedDTW(this.transformType);
    }

     @Override
    public String getParameters() {
        return super.getParameters()+",transformType,"+this.transformType;
    }
    
    public static class TransformWeightedDTW extends DD_DTW.GoreckiDerivativesDTW{
        
        private TransformType transformType;
        
        public TransformWeightedDTW(TransformType transformType){
            super();
            this.transformType = transformType;
        }
        
        public double[] getNonScaledDistances(Instance first, Instance second){

            DTW_DistanceBasic dtw = new DTW_DistanceBasic();
            int classPenalty = 0;
            if(first.classIndex()>0){
                classPenalty=1;
            }
            Instances temp = new Instances(first.dataset(),0);
            temp.add(first);
            temp.add(second);
            try{
                switch(this.transformType){
                    case COS:
                        temp = new Cosine().process(temp);
                        break;
                    case SIN:
                        temp = new Sine().process(temp);
                        break;
                    case HIL:
                        temp = new Hilbert().process(temp);
                        break;
                }
            }catch(Exception e){
                e.printStackTrace();
                return null;
            }        

            double dist = dtw.distance(first, second);
            double transDist = dtw.distance(temp.get(0), temp.get(1), Double.MAX_VALUE);

            return new double[]{Math.sqrt(dist),Math.sqrt(transDist)};
        }
        
    }
    
    
    public static void recreateResultsTable() throws Exception{
        System.out.println("Recreating Results from Gorecki 2:");
        Instances train, test;
        DTW_DistanceBasic dtw;
        kNN knn;
        double acc, err;
        int correct;
        DecimalFormat df = new DecimalFormat("#.##");
        Instances transTrain, transTest;
        
        SimpleBatchFilter[] transforms = {new Cosine(), new Sine(), new Hilbert()};
        TransformType[] transformTypes = {TransformType.COS,TransformType.SIN,TransformType.HIL};
        System.out.println("Dataset,fullCosDTW,fullSinDTW,fullHilDTW,weightedCosDTW,weightedSinDTW,weightedHilDTW");
        for(String dataset:PAPER_DATASETS){
            System.out.print(dataset+",");
            train = ClassifierTools.loadData(DATA_DIR+dataset+"/"+dataset+"_TRAIN");
            test = ClassifierTools.loadData(DATA_DIR+dataset+"/"+dataset+"_TEST");
            
            // DTW on only the transformed data first
            for(SimpleBatchFilter transform:transforms){
                transTrain = transform.process(train);
                transTest = transform.process(test);
                dtw = new DTW_DistanceBasic();
                knn = new kNN();
                knn.setDistanceFunction(dtw);
                correct = getCorrect(knn, transTrain, transTest);
                acc = (double)correct/test.numInstances();
                err = (1-acc)*100;
                System.out.print(df.format(err)+",");
            }
            
            // now use a combination of the raw and transform
            for(TransformType transform:transformTypes){
                DTD_C tdtw = new DTD_C(transform);
                correct = getCorrect(tdtw, train, test);
                acc = (double)correct/test.numInstances();
                err = (1-acc)*100;
                System.out.print(df.format(err)+",");
            }
            System.out.println("");
        }
    }
    
    public static void main(String[] args){
        
        // option 1: simple example of the classifier
        // option 2: recreate the results from the original published work
        
        int option = 1;
        
        try{
            if(option==1){
                String dataName = "ItalyPowerDemand";
                Instances train = ClassifierTools.loadData(DATA_DIR+dataName+"/"+dataName+"_TRAIN");
                Instances test = ClassifierTools.loadData(DATA_DIR+dataName+"/"+dataName+"_TEST");
                
                // create the classifier, using cosine in the distance calculations as an example
                DTD_C nntw = new DTD_C(TransformType.COS);
                
                // params a and b have not been explicitly set, so buildClassifier will cv to find them
                nntw.buildClassifier(train);
                
                int correct = 0;
                for(int i = 0; i < test.numInstances(); i++){
                    if(nntw.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                        correct++;
                    }
                }
                System.out.println(dataName+":\t"+new DecimalFormat("#.###").format((double)correct/test.numInstances()*100)+"%");
                
            }else if(option==2){
                recreateResultsTable();
            }
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}
