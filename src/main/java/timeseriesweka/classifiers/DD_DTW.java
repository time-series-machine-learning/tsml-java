package timeseriesweka.classifiers;

import java.text.DecimalFormat;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.classifiers.lazy.kNN;
import utilities.ClassifierResults;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import timeseriesweka.elastic_distance_measures.DTW_DistanceBasic;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 * 
 * Implementation of:
 * Górecki, Tomasz, and Maciej Łuczak. 
 * Using derivatives in time series classification.
 * Data Mining and Knowledge Discovery 26.2 (2013): 310-331.
 * 
 */

/*
=========================================================================================================================
BRIEF DESCRIPTION:
=========================================================================================================================

The classifier works by using a weighted combination of the raw data representation and derivatives with either
Euclidean Distance or full-window DTW. The idea is that the distance of using the distance measure on the raw 
data is weighted using parameter a, and the distance using derivative-transformed data is weighted using a 
parameter b. These two params are in the range of 0-1, and the sum of a+b is always 1 (i.e. binary split in the 
weighting between the two classifiers. Therefore a = 1, b = 0 is equivilent to just using DTW, and a=0, b=1
is equivilent to just using derivative DTW. 

The author's propose using a single parameter alpha to weight these two components by using it to derive values of 
a and b. However, in our experiments this approach does not seem to reproduce the published results. However, simply
searching from a = 0 to 1 and b = 1 to 0 in increments of 0.01 (101 param options) appears to reproduce results

=========================================================================================================================
HOW TO USE:
=========================================================================================================================
The class extends the kNN class, so classifier functionality is included. Three parameters should be set: 
1. whether the classifier uses ED or DTW (default is ED unless set) (handled by enum in constructor)
2. values for a and b (defaults to a=1 and b=0, equiv to just ED or DTW) 

The params a and b can be set explicitly through a mutator. However, if not specified, the buildClassifier method
performs the LOOCV procedure outlined in the original paper to find the values of a and b using the training data.

=========================================================================================================================
RECREATING RESULTS:
=========================================================================================================================
See method recreateResultsTable()
Data dir of the TSC problems, DATA_DIR,  must be set to match local implementation

=========================================================================================================================
RELATED CLASSES:
=========================================================================================================================
Next iteration:
NNTranformWeighting.java 
*/

public class DD_DTW extends kNN implements SaveParameterInfo{
     protected ClassifierResults res =new ClassifierResults();

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "T. Górecki and M. Łuczak");
        result.setValue(TechnicalInformation.Field.TITLE, "Using derivatives in time series classification");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.VOLUME, "26");
        result.setValue(TechnicalInformation.Field.NUMBER,"2");
        result.setValue(TechnicalInformation.Field.PAGES, "310-331");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        return result;
    }    
    
    public static final String DATA_DIR = "C:/Temp/Dropbox/TSC Problems/";
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
    public static final String[] GORECKI_DATASETS = {
        //<editor-fold defaultstate="collapsed" desc="Datasets from the paper">
        "fiftywords", // 450,455,270,50
        "Adiac", // 390,391,176,37
        "Beef", // 30,30,470,5
        "CBF", // 30,900,128,3
        "Coffee", // 28,28,286,2
        "FaceAll", // 560,1690,131,14
        "FaceFour", // 24,88,350,4
        "fish", // 175,175,463,7
        "GunPoint", // 50,150,150,2
        "Lightning2", // 60,61,637,2
        "Lightning7", // 70,73,319,7
        "OliveOil", // 30,30,570,4
        "OSULeaf", // 200,242,427,6
        "SwedishLeaf", // 500,625,128,15
        "SyntheticControl", // 300,300,60,6
        "Trace", // 100,100,275,4
        "TwoPatterns", // 1000,4000,128,4
        "wafer", // 1000,6164,152,2
        "yoga"// 300,3000,426,2
        //</editor-fold>
    };
    protected GoreckiDerivativesEuclideanDistance distanceFunction;
    protected boolean paramsSet;
    protected boolean sampleForCV=false;
    protected double prop;
    public void sampleForCV(boolean b, double p){
        sampleForCV=b;
        prop=p;
    }
    
    public enum DistanceType{EUCLIDEAN, DTW};
    
    // defaults to Euclidean distance
    public DD_DTW(){
        super();
        this.distanceFunction = new GoreckiDerivativesDTW();
        this.paramsSet = false;
        
    }
    
    public DD_DTW(DistanceType distType){
        super();
        if(distType==DistanceType.EUCLIDEAN){
            this.distanceFunction = new GoreckiDerivativesEuclideanDistance();
        }else{
            this.distanceFunction = new GoreckiDerivativesDTW();
        }
        this.paramsSet = false;
    }
    
    public void setAandB(double a, double b){
        this.distanceFunction.a = a;
        this.distanceFunction.b = b;
        this.paramsSet = true;
    }
    
    @Override
    public void buildClassifier(Instances train){
        res.buildTime=System.currentTimeMillis();
        
        if(!paramsSet){
            this.distanceFunction.crossValidateForAandB(train);
            paramsSet=true;
        }
        this.setDistanceFunction(this.distanceFunction);
        super.buildClassifier(train);
        res.buildTime=System.currentTimeMillis()-res.buildTime;
    }
     @Override
    public String getParameters() {
        return "BuildTime,"+res.buildTime+",a,"+distanceFunction.a+",b,"+distanceFunction.b;
    }
     
   
    
    public static class GoreckiDerivativesEuclideanDistance extends EuclideanDistance{
    
        protected double alpha;
        protected double a;
        protected double b;
        public boolean sampleTrain=true;    //Change back to default to false

        public GoreckiDerivativesEuclideanDistance(){
            this.a = 1;
            this.b = 0;
            this.alpha = -1;
            // defaults to no derivative input
        }
        public GoreckiDerivativesEuclideanDistance(Instances train){
            // this is what the paper suggests they use, but doesn't reproduce results. 
            //this.crossValidateForAlpha(train);

            // when cv'ing for a = 0:0.01:1 and b = 1:-0.01:0 results can be reproduced though, so use that
            this.crossValidateForAandB(train);
        }
        public GoreckiDerivativesEuclideanDistance(double alpha){
            this.alpha = alpha;
            this.a = Math.cos(alpha);
            this.b = Math.sin(alpha);
        }

        public GoreckiDerivativesEuclideanDistance(double a, double b){
            this.alpha = alpha;
            this.a = Math.cos(alpha);
            this.b = Math.sin(alpha);
        }

        @Override
        public double distance(Instance one, Instance two){
            return this.distance(one, two, Double.MAX_VALUE);
        }

        @Override
        public double distance(Instance one, Instance two, double cutoff, PerformanceStats stats){
            return this.distance(one,two,cutoff);
        }

        @Override
        public double distance(Instance first, Instance second, double cutoff){
            double dist = 0;
            double dirDist = 0;

            int classPenalty = 0;
            if(first.classIndex()>0){
                classPenalty=1;
            }

            double firstDir, secondDir;
            for(int i = 0; i < first.numAttributes()-classPenalty; i++){
                dist+= ((first.value(i)-second.value(i))*(first.value(i)-second.value(i)));

                // one less for derivatives, since we don't want to include the class value!
                // could skip the first instead of last, but this makes more sense for earlier early abandon
                if(i < first.numAttributes()-classPenalty-1){
                    firstDir = first.value(i+1)-first.value(i);
                    secondDir = second.value(i+1)-second.value(i);
                    dirDist+= ((firstDir-secondDir)*(firstDir-secondDir));
                }

            }
            return(a*Math.sqrt(dist)+b*Math.sqrt(dirDist));
        }

        public double[] getNonScaledDistances(Instance first, Instance second){
            double dist = 0;
            double dirDist = 0;

            int classPenalty = 0;
            if(first.classIndex()>0){
                classPenalty=1;
            }

            double firstDir, secondDir;

            for(int i = 0; i < first.numAttributes()-classPenalty; i++){
                dist+= ((first.value(i)-second.value(i))*(first.value(i)-second.value(i)));

                if(i < first.numAttributes()-classPenalty-1){
                    firstDir = first.value(i+1)-first.value(i);
                    secondDir = second.value(i+1)-second.value(i);
                    dirDist+= ((firstDir-secondDir)*(firstDir-secondDir));
                }
            }
            return new double[]{Math.sqrt(dist),Math.sqrt(dirDist)};
        }

        // implemented to mirror original MATLAB implementeation that's described in the paper (with appropriate modifications)
        public double crossValidateForAlpha(Instances tr){
            Instances train=tr;
            if(sampleTrain){
                tr=InstanceTools.subSample(tr, tr.numInstances()/10, 0);
            }
            
            double[] labels = new double[train.numInstances()];
            for(int i = 0; i < train.numInstances(); i++){
                labels[i] = train.instance(i).classValue();
            }

            double[] a = new double[ALPHAS.length];
            double[] b = new double[ALPHAS.length];

            for(int alphaId = 0; alphaId < ALPHAS.length; alphaId++){
                a[alphaId] = Math.cos(ALPHAS[alphaId]);
                b[alphaId] = Math.sin(ALPHAS[alphaId]);
            }

            int n = train.numInstances();
            int k = ALPHAS.length;
            int[] mistakes = new int[k];
    //
    //            // need to get the derivatives (MATLAB code uses internal diff function instead)
    //            Instances dTrain = new GoreckiDerivativesDistance.GoreckiDerivativeFilter().process(train);

            double[] D;
            double[] L;
            double[] d;
            double dist;
            double dDist;

            double[] individualDistances;

            for(int i = 0; i < n; i++){

                D = new double[k];
                L = new double[k];
                for(int j = 0; j < k; j++){
                    D[j]=Double.MAX_VALUE;
                }

                for(int j = 0; j < n; j++){
                    if(i==j){
                        continue;
                    }

                    individualDistances = this.getNonScaledDistances(train.instance(i), train.instance(j));
                    // have to be a bit different here, since we can't vectorise in Java
    //                    dist = distanceFunction.distance(train.instance(i), train.instance(j));
    //                    dDist = distanceFunction.distance(dTrain.instance(i), dTrain.instance(j));
                    dist = individualDistances[0];
                    dDist = individualDistances[1];

                    d = new double[k];

                    for(int alphaId = 0; alphaId < k; alphaId++){
                        d[alphaId] = a[alphaId]*dist+b[alphaId]*dDist;
                        if(d[alphaId] < D[alphaId]){
                            D[alphaId]=d[alphaId];
                            L[alphaId]=labels[j];
                        }
                    }
                }

                for(int alphaId = 0; alphaId < k; alphaId++){
                    if(L[alphaId]!=labels[i]){
                        mistakes[alphaId]++;
                    }
                }
            }

            int bsfMistakes = Integer.MAX_VALUE;
            int bsfAlphaId = -1;
            for(int alpha = 0; alpha < k; alpha++){
                if(mistakes[alpha] < bsfMistakes){
                    bsfMistakes = mistakes[alpha];
                    bsfAlphaId = alpha;
                }
            }
            this.alpha = ALPHAS[bsfAlphaId];
            this.a = Math.cos(this.alpha);
            this.b = Math.sin(this.alpha);
    //        System.out.println("bestAlphaId,"+bsfAlphaId);
            return (double)(train.numInstances()-bsfMistakes)/train.numInstances();
        }

        // changed to now return the predictions of the best alpha parameter
        public double[] crossValidateForAandB(Instances tr){
            Instances train=tr;
            if(sampleTrain){
                tr=InstanceTools.subSample(tr, tr.numInstances()/10, 0);
            }
            
            double[] labels = new double[train.numInstances()];
            for(int i = 0; i < train.numInstances(); i++){
                labels[i] = train.instance(i).classValue();
            }

            double[] a = new double[101];
            double[] b = new double[101];

            for(int alphaId = 0; alphaId <= 100; alphaId++){
                a[alphaId] = (double)(100-alphaId)/100;
                b[alphaId] = (double)alphaId/100;
            }

            int n = train.numInstances();
            int k = a.length;
            int[] mistakes = new int[k];

            double[] D;
            double[] L;
            double[] d;
            double dist;
            double dDist;

            double[][] LforAll = new double[n][];

            double[] individualDistances;

            for(int i = 0; i < n; i++){

                D = new double[k];
                L = new double[k];
                for(int j = 0; j < k; j++){
                    D[j]=Double.MAX_VALUE;
                }

                for(int j = 0; j < n; j++){
                    if(i==j){
                        continue;
                    }

                    individualDistances = this.getNonScaledDistances(train.instance(i), train.instance(j));
                    dist = individualDistances[0];
                    dDist = individualDistances[1];

                    d = new double[k];

                    for(int alphaId = 0; alphaId < k; alphaId++){
                        d[alphaId] = a[alphaId]*dist+b[alphaId]*dDist;
                        if(d[alphaId] < D[alphaId]){
                            D[alphaId]=d[alphaId];
                            L[alphaId]=labels[j];
                        }
                    }
                }

                for(int alphaId = 0; alphaId < k; alphaId++){
                    if(L[alphaId]!=labels[i]){
                        mistakes[alphaId]++;
                    }
                }
                LforAll[i] = L;
            }

            int bsfMistakes = Integer.MAX_VALUE;
            int bsfAlphaId = -1;
            for(int alpha = 0; alpha < k; alpha++){
                if(mistakes[alpha] < bsfMistakes){
                    bsfMistakes = mistakes[alpha];
                    bsfAlphaId = alpha;
                }
            }

            this.alpha = -1;
            this.a = a[bsfAlphaId];
            this.b = b[bsfAlphaId];
            double[] bestAlphaPredictions = new double[train.numInstances()];
            for(int i = 0; i < bestAlphaPredictions.length; i++){
                bestAlphaPredictions[i] = LforAll[i][bsfAlphaId];
            }
            return bestAlphaPredictions;
        }

        public double getA() {
            return a;
        }

        public double getB() {
            return b;
        }


    }

    public static class GoreckiDerivativesDTW extends GoreckiDerivativesEuclideanDistance{

        public GoreckiDerivativesDTW(){
            super();
        }
        public GoreckiDerivativesDTW(Instances train){
            super(train);
        }
        public GoreckiDerivativesDTW(double alpha){
            super(alpha);
        }

        public GoreckiDerivativesDTW(double a, double b){
            super(a,b);
        }

        @Override
        public double distance(Instance one, Instance two){
            return this.distance(one, two, Double.MAX_VALUE);
        }

        @Override
        public double distance(Instance one, Instance two, double cutoff, PerformanceStats stats){
            return this.distance(one,two,cutoff);
        }

        @Override
        public double distance(Instance first, Instance second, double cutoff){

            double[] distances = getNonScaledDistances(first, second);
            return a*distances[0]+b*distances[1];
        }

        public double[] getNonScaledDistances(Instance first, Instance second){
            double dist = 0;
            double derDist = 0;

    //        DTW dtw = new DTW();
            DTW_DistanceBasic dtw = new DTW_DistanceBasic();
            int classPenalty = 0;
            if(first.classIndex()>0){
                classPenalty=1;
            }

            GoreckiDerivativeFilter filter = new GoreckiDerivativeFilter();
            Instances temp = new Instances(first.dataset(),0);
            temp.add(first);
            temp.add(second);
            try{
                temp = filter.process(temp);
            }catch(Exception e){
                e.printStackTrace();
                return null;
            }        

            dist = dtw.distance(first, second);
            derDist = dtw.distance(temp.get(0), temp.get(1), Double.MAX_VALUE);

            return new double[]{Math.sqrt(dist),Math.sqrt(derDist)};
        }


    }

    // They calculate derivatives differently to the transform we have (which matches Keogh et al.'s DDTW implemetation)
    // Derivatives are built into the new distance measures, but this is needed to recreating the derivative Euclidean/DTW comparison results 
    private static class GoreckiDerivativeFilter extends weka.filters.SimpleBatchFilter{

        @Override
        public String globalInfo() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        protected Instances determineOutputFormat(Instances inputFormat) throws Exception {

            Instances output = new Instances(inputFormat,0);
            output.deleteAttributeAt(0);
            output.setRelationName("goreckiDerivative_"+output.relationName());
            for(int a = 0; a < output.numAttributes()-1; a++){
                output.renameAttribute(a, "derivative_"+a);
            }

            return output;

        }

        @Override
        public Instances process(Instances instances) throws Exception {

            Instances output = determineOutputFormat(instances);
            Instance thisInstance;
            Instance toAdd;
            double der;
            for(int i = 0; i < instances.numInstances(); i++){
                thisInstance = instances.get(i);
                toAdd = new DenseInstance(output.numAttributes());
                for(int a = 1; a < instances.numAttributes()-1; a++){
                    der = thisInstance.value(a)-thisInstance.value(a-1);
                    toAdd.setValue(a-1, der);
                }
                toAdd.setValue(output.numAttributes()-1, thisInstance.classValue());
                output.add(toAdd);
            }
            return output;
        }

    }


    public static void recreateResultsTable() throws Exception{
        recreateResultsTable(0);
    }


    public static void recreateResultsTable(int seed) throws Exception{

        String[] datasets = GORECKI_DATASETS;

        String dataDir = "C:/Temp/Dropbox/TSC Problems/";
        Instances train, test, dTrain, dTest;
        EuclideanDistance ed;
        kNN knn;
        int correct;
        double acc, err;
        DecimalFormat df = new DecimalFormat("##.##");

        // important - use the correct one! Gorecki uses different derivatives to Keogh
        GoreckiDerivativeFilter derFilter = new GoreckiDerivativeFilter();

        StringBuilder st = new StringBuilder();
        System.out.println("Dataset,ED,DED,DD_ED,DTW,DDTW,DD_DTW");


        for(String dataset:datasets){

            System.out.print(dataset+",");

            train = ClassifierTools.loadData(dataDir+dataset+"/"+dataset+"_TRAIN");
            test = ClassifierTools.loadData(dataDir+dataset+"/"+dataset+"_TEST");

            // instance resampling happens here, seed of 0 means that the standard train/test split is used
            if(seed!=0){
                Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, seed);
                train = temp[0];
                test = temp[1];
            }

            dTrain = derFilter.process(train);
            dTest = derFilter.process(test);

            // ED 
//            ed = new GoreckiEuclideanDistance();
            ed = new EuclideanDistance();
            ed.setDontNormalize(true);
            knn = new kNN(ed);
            correct = getCorrect(knn, train, test);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            System.out.print(df.format(err)+",");

            // DED
            ed = new EuclideanDistance();
            knn = new kNN(ed);
            correct = getCorrect(knn, dTrain, dTest);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            System.out.print(df.format(err)+",");

            //DD_ED
            DD_DTW dd_ed = new DD_DTW(DistanceType.EUCLIDEAN);
            correct = getCorrect(dd_ed, train, test);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            System.out.print(df.format(err)+",");

            //DTW
            DTW_DistanceBasic dtw = new DTW_DistanceBasic();
            knn = new kNN(dtw);
            correct = getCorrect(knn, train, test);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            System.out.print(df.format(err)+",");

            // DDTW
            DTW_DistanceBasic dDtw = new DTW_DistanceBasic();
            knn = new kNN(dDtw);
            correct = getCorrect(knn, dTrain, dTest);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            System.out.print(df.format(err)+",");

            // DDDTW
            DD_DTW dd_dtw = new DD_DTW(DistanceType.DTW);
            correct = getCorrect(dd_dtw, train, test);
            acc = (double)correct/test.numInstances();
            err = (1-acc)*100;
            System.out.println(df.format(err));
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
                
                // create the classifier, using DTW as the distance function as an example
                DD_DTW nndw = new DD_DTW(DistanceType.DTW);;
                
                // params a and b have not been explicitly set, so buildClassifier will cv to find them
                nndw.buildClassifier(train);
                
                int correct = 0;
                for(int i = 0; i < test.numInstances(); i++){
                    if(nndw.classifyInstance(test.instance(i))==test.instance(i).classValue()){
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








    protected static int getCorrect(kNN knn, Instances train, Instances test) throws Exception{
        knn.buildClassifier(train);
        int correct = 0;
        for(int i = 0; i < test.numInstances(); i++){
            if(test.instance(i).classValue()==knn.classifyInstance(test.instance(i))){
                correct++;
            }
        }
        return correct;
    }
    
    
    
}
