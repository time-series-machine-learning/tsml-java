package timeseriesweka.classifiers.ensembles.elastic_ensemble;

import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.elastic_distance_measures.DTW;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class DTW1NN extends Efficient1NN {

    
    private double r = 1;
    
    /**
     * Constructor with specified window size (between 0 and 1). When a window
     * size is specified, cross-validation methods will become inactive for this
     * object. 
     * 
     * Note: if window = 1, classifierIdentifier will be DTW_R1_1NN; other 
     * window sizes will results in cId of DTW_Rn_1NN instead. This information
     * is used for any file writing
     * @param r 
     */
    public DTW1NN(double r){
        this.allowLoocv = false;
        this.r = r;
        if(r!=1){
            this.classifierIdentifier = "DTW_Rn_1NN";
        }else{
            this.classifierIdentifier = "DTW_R1_1NN";
        }
    }

    /**
     * A default constructor. Sets the window to 1 (100%), but allows for the 
     * option of cross-validation if the relevant method is called. 
     * 
     * classifierIdentifier is initially set to DTW_R1_1NN, but will 
     * update automatically to DTW_Rn_1NN if loocv is called
     */
    public DTW1NN(){
        this.r = 1;
        this.classifierIdentifier = "DTW_R1_1NN";
    }
    public void setWindow(double w){ r=w;}
    
    public void turnOffCV(){
        this.allowLoocv = false;
    }
     public void turnOnCV(){
        this.allowLoocv = true;
    }
   
    @Override
    public double[] loocv(Instances train) throws Exception{
        if(this.allowLoocv==true && this.classifierIdentifier.contains("R1")){
            this.classifierIdentifier=this.classifierIdentifier.replace("R1", "Rn");
        }
        return super.loocv(train);
    }

    @Override
    public double[] loocv(Instances[] train) throws Exception{
        if(this.allowLoocv==true && this.classifierIdentifier.contains("R1")){
            this.classifierIdentifier=this.classifierIdentifier.replace("R1", "Rn");
        }
        return super.loocv(train);
    }
            
    
    final public int getWindowSize(int n){
        int w=(int)(r*n);   //Rounded down.
                //No Warp, windowSize=1
        if(w<1) w=1;
                //Full Warp : windowSize=n, otherwise scale between		
        else if(w<n)    
                w++;
        return w;	
    }
    
    
    
//    public double classifyInstance(Instance instance) throws Exception{
//        if(isDerivative){
//            Instances temp = new Instances(instance.dataset(),1);
//            temp.add(instance);
//            temp = new DerivativeFilter().process(temp);
//            return classifyInstance(temp.instance(0));
//        }
//        return super.classifyInstance(instance);
//    }
    
    public final double distance(Instance first, Instance second, double cutoff){
        
        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
        if(first.classIndex() != first.numAttributes()-1 || second.classIndex()!=second.numAttributes()-1){
            DTW temp = new DTW();
            temp.setR(r);
            return temp.distance(first, second,cutoff);
        }        
        
        double minDist;
        boolean tooBig;

        int n = first.numAttributes()-1;
        int m = second.numAttributes()-1;
        /*  Parameter 0<=r<=1. 0 == no warp, 1 == full warp 
         generalised for variable window size
         * */
        int windowSize = getWindowSize(n);
//Extra memory than required, could limit to windowsize,
//        but avoids having to recreate during CV 
//for varying window sizes        
        double[][] matrixD = new double[n][m];
        
        /*
         //Set boundary elements to max. 
         */
        int start, end;
        for (int i = 0; i < n; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = i + windowSize + 1 < m ? i + windowSize + 1 : m;
            for (int j = start; j < end; j++) {
                matrixD[i][j] = Double.MAX_VALUE;
            }
        }
        matrixD[0][0] = (first.value(0) - second.value(0)) * (first.value(0) - second.value(0));
//a is the longer series. 
//Base cases for warping 0 to all with max interval	r	
//Warp first.value(0] onto all second.value(1]...second.value(r+1]
        for (int j = 1; j < windowSize && j < m; j++) {
            matrixD[0][j] = matrixD[0][j - 1] + (first.value(0) - second.value(j)) * (first.value(0) - second.value(j));
        }

//	Warp second.value(0] onto all first.value(1]...first.value(r+1]
        for (int i = 1; i < windowSize && i < n; i++) {
            matrixD[i][0] = matrixD[i - 1][0] + (first.value(i) - second.value(0)) * (first.value(i) - second.value(0));
        }
//Warp the rest,
        for (int i = 1; i < n; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = i + windowSize < m ? i + windowSize : m;
            for (int j = start; j < end; j++) {
                minDist = matrixD[i][j - 1];
                if (matrixD[i - 1][j] < minDist) {
                    minDist = matrixD[i - 1][j];
                }
                if (matrixD[i - 1][j - 1] < minDist) {
                    minDist = matrixD[i - 1][j - 1];
                }
                matrixD[i][j] = minDist + (first.value(i) - second.value(j)) * (first.value(i) - second.value(j));
                if (tooBig && matrixD[i][j] < cutoff) {
                    tooBig = false;
                }
            }
            //Early abandon
            if (tooBig) {
                return Double.MAX_VALUE;
            }
        }
//Find the minimum distance at the end points, within the warping window. 
        return matrixD[n-1][m-1];
    }
    
    
    
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
       double[] res=new double[instance.numClasses()];
       int r=(int)classifyInstance(instance);
       res[r]=1;
       return res;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void runComparison() throws Exception{
        String tscProbDir = "C:/users/sjx07ngu/Dropbox/TSC Problems/";
        
//        String datasetName = "ItalyPowerDemand";
        String datasetName = "GunPoint";
//        String datasetName = "Beef";
//        String datasetName = "Coffee";
//        String datasetName = "SonyAiboRobotSurface1";

        double r = 0.1;
        Instances train = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TEST");
        
        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        DTW oldDtw = new DTW();
        oldDtw.setR(r);
        knn.setDistanceFunction(oldDtw);
        knn.buildClassifier(train);
        
        // new version
        DTW1NN dtwNew = new DTW1NN(r);
        dtwNew.buildClassifier(train);
        
        int correctOld = 0;
        int correctNew = 0;
        
        long start, end, oldTime, newTime;
        double pred;
               
        // classification with old MSM class and kNN
        start = System.nanoTime();
        
        correctOld = 0;
        for(int i = 0; i < test.numInstances(); i++){
            pred = knn.classifyInstance(test.instance(i));
            if(pred==test.instance(i).classValue()){
                correctOld++;
            }
        }
        end = System.nanoTime();
        oldTime = end-start;
        
        // classification with new MSM and own 1NN
        start = System.nanoTime();
        correctNew = 0;
        for(int i = 0; i < test.numInstances(); i++){
            pred = dtwNew.classifyInstance(test.instance(i));
            if(pred==test.instance(i).classValue()){
                correctNew++;
            }
        }
        end = System.nanoTime();
        newTime = end-start;
        
        System.out.println("Comparison of MSM: "+datasetName);
        System.out.println("==========================================");
        System.out.println("Old acc:    "+((double)correctOld/test.numInstances()));
        System.out.println("New acc:    "+((double)correctNew/test.numInstances()));
        System.out.println("Old timing: "+oldTime);
        System.out.println("New timing: "+newTime);
        System.out.println("Relative Performance: " + ((double)newTime/oldTime));
    }
    
      
    public static void main(String[] args) throws Exception{
        for(int i = 0; i < 10; i++){
            runComparison();
        }
    }

    @Override
    public void setParamsFromParamId(Instances train, int paramId) {
        if(this.allowLoocv){
            if(this.classifierIdentifier.contains("R1")){
                this.classifierIdentifier=this.classifierIdentifier.replace("R1", "Rn");
            }
            this.r = (double)paramId/100;
        }else{
            throw new RuntimeException("Warning: trying to set parameters of a fixed window DTW");
        }
    }

    @Override
    public String getParamInformationString() {
        return this.r+"";
    }


}
