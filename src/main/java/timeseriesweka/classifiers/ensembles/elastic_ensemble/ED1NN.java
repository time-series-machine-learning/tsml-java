// adjusted April '16
// note: not using DTW class in here (redoing the method) as even though the DTW class is already about as efficient, it still
// involves some array copying. Here we can opperate straight on the Instance values instead

package timeseriesweka.classifiers.ensembles.elastic_ensemble;

import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.elastic_distance_measures.DTW;
//import efficient_standalone_classifiers.Eff
/**
 *
 * @author sjx07ngu
 */
public class ED1NN extends Efficient1NN{
    
    public ED1NN(){
        this.classifierIdentifier = "Euclidean_1NN";
        this.allowLoocv = false;
        this.singleParamCv = true;
    }

    public final double distance(Instance first, Instance second, double cutoff){
        
        // base case - we're assuming class val is last. If this is true, this method is fine,
        // if not, we'll default to the DTW class
        if(first.classIndex() != first.numAttributes()-1 || second.classIndex()!=second.numAttributes()-1){
            EuclideanDistance temp = new EuclideanDistance();
            temp.setDontNormalize(true);
            return temp.distance(first, second,cutoff);
        }        
         
        double sum = 0;
        for(int a = 0; a < first.numAttributes()-1;a++){
            sum += (first.value(a)-second.value(a))*(first.value(a)-second.value(a));
        }
        
//        return Math.sqrt(sum);
        return sum;
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] dist=new double[instance.numClasses()];
        dist[(int)classifyInstance(instance)]=1;
        return dist;
                }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void runComparison() throws Exception{
        String tscProbDir = "C:/users/sjx07ngu/Dropbox/TSC Problems/";
        
//        String datasetName = "ItalyPowerDemand";
//        String datasetName = "GunPoint";
//        String datasetName = "Beef";
//        String datasetName = "Coffee";
        String datasetName = "SonyAiboRobotSurface1";

        double r = 0.1;
        Instances train = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData(tscProbDir+datasetName+"/"+datasetName+"_TEST");
        
        // old version
        kNN knn = new kNN(); //efaults to k = 1 without any normalisation
        EuclideanDistance oldED = new EuclideanDistance();
        oldED.setDontNormalize(true);
        knn.setDistanceFunction(oldED);
        knn.buildClassifier(train);
        
        // new version
        ED1NN edNew = new ED1NN();
        edNew.buildClassifier(train);
        
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
            pred = edNew.classifyInstance(test.instance(i));
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
        // could throw an Exception but it shouldn't make a difference since this measure has no params, 
        // so just warn the user that they're probably not doing what they think they are!
//        System.err.println("warning: ED has not parameters to set; call to setParamFromParamId made no changes");
    }

    @Override
    public String getParamInformationString() {
        return "NoParams";
    }


}
