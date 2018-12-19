/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multivariate_timeseriesweka.classifiers;

import multivariate_timeseriesweka.measures.DTW_D;
import multivariate_timeseriesweka.measures.DTW_I;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import utilities.generic_storage.Pair;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.elastic_distance_measures.DTW_DistanceBasic;
import static utilities.InstanceTools.findMinDistance;

/**
 *
 * @author ABostrom
 */
public class NN_DTW_A extends AbstractClassifier{

    Instances train;
    
    public double threshold;
       
    DTW_DistanceBasic I;
    DTW_DistanceBasic D;
    
    double R;
    
    public NN_DTW_A(){
        I = new DTW_I();
        D = new DTW_D();
    }
    
    public void setR(double r){
        R = r;
        I.setR(R);
        D.setR(R);
    }
    
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        return result;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception{
        train = data;
        threshold = learnThreshold(train);
        System.out.println("threshold = " + threshold);
        //build DTW_A. doesn't matter what function it uses for building as its' lazy.
        //default A to support a distance function of some kind.
    }   
    
    @Override
    public double classifyInstance(Instance instance) throws Exception{
        Pair<Instance, Double> minD = findMinDistance(train, instance, D);
        Pair<Instance, Double> minI = findMinDistance(train, instance, I);
        //System.out.println("minD = " + minD + "minI = " + minI);
        double S =  minD.var2 / (minI.var2 + 0.000000001);
        double out = S > threshold ? minI.var1.classValue() : minD.var1.classValue();
        
        //System.out.println("minD " + minD.var2 + " minI "+ minI.var2 + " S " + S);
        return out;
    }
    
    public double learnThreshold(Instances data){
        Pair<List<Double>, List<Double>> scores = findScores(data);
        List<Double> S_dSuccess = scores.var1;
        List<Double> S_iSuccess = scores.var2;
        
        double output;
        if(S_iSuccess.isEmpty() && S_dSuccess.isEmpty())
            output= 1;
        else if(!S_iSuccess.isEmpty() && S_dSuccess.isEmpty())
            output = Collections.min(S_iSuccess) -0.1; //they take off 0.1
        else if(S_iSuccess.isEmpty() && !S_dSuccess.isEmpty())
            output = Collections.max(S_dSuccess) + 0.1; //they add on 0.1
        else
            output = calculateThreshold(S_dSuccess, S_iSuccess); 
            
        return output;
    }
    
    double calculateThreshold(List<Double> dSuccess, List<Double> iSuccess){
        double output = 0;
        //trying to minimse common
        int common = iSuccess.size() + dSuccess.size();           
        for (int j = 0;j<dSuccess.size();j++){
            int in = 0;
            int dp = 0;
            for (int i = 0;i<dSuccess.size();i++){
                if (dSuccess.get(i) >= dSuccess.get(j)){
                    dp++;
                }    
            }

            for (int i = 0;i<iSuccess.size();i++){
                if (iSuccess.get(i) < dSuccess.get(j)){
                    in++;
                }    
            }

            if (in+dp < common){
                common = in+dp;
                output = dSuccess.get(j);
            }
        }
            
        for (int j = 0; j<iSuccess.size();j++){
            int in = 0;
            int dp = 0;
            for (int i = 0;i<dSuccess.size();i++){
                if (dSuccess.get(i) >= iSuccess.get(j)){
                    dp++;
                }    
            }

            for (int i = 0;i<iSuccess.size();i++){
                if (iSuccess.get(i) < iSuccess.get(j)){
                    in++;
                }    
            }

            if (in+dp < common){
                common = in+dp;
                output = iSuccess.get(j);
            }
        }
            
        return output;
    }

    
    Pair<List<Double>, List<Double>> findScores(Instances data){
        List<Double> S_dSuccess = new ArrayList<>();
        List<Double> S_iSuccess = new ArrayList<>();
        
        for(int i=0; i<data.numInstances(); i++){
            try {
                //LOOCV search for distances.
                Instances cv_train = data.trainCV(data.numInstances(), i);
                Instances cv_test = data.testCV(data.numInstances(), i);
                Instance test = cv_test.firstInstance();
                
                Pair<Instance, Double> pair_D = findMinDistance(cv_train, test, D);
                Pair<Instance, Double> pair_I = findMinDistance(cv_train, test, I);
                
                //we know we only have one instance.
                double pred_d = pair_D.var1.classValue();
                double pred_i = pair_I.var1.classValue();
                double dist_d = pair_D.var2;
                double dist_i = pair_I.var2;
                double S = dist_d / (dist_i+0.000000001);
                
                //if d is correct and i is incorrect.
                if(test.classValue() == pred_d && test.classValue() != pred_i)
                    S_dSuccess.add(S);
                //if d is incorrect and i is correct.
                if(test.classValue() != pred_d && test.classValue() == pred_i)
                    S_iSuccess.add(S);
            } catch (Exception ex) {
                System.out.println(ex);
            }
            
        }
       
        return new Pair(S_dSuccess, S_iSuccess);
    }
    

    
    @Override
    public String toString(){
        return "threshold="+threshold;
    }

}
