package weka.classifiers.meta;

import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.*;

/**
 *
 * @author ajb
 */
public class HeterogeneousEnsemble extends AbstractClassifier{
//The classifiers MUST be added externally
    ArrayList<Classifier> ensemble;
    Instances train;
    double[] weights;
    double weightsSD=0;
    public enum WeightType{EQUAL,CV}
    WeightType w;
    private HeterogeneousEnsemble(){
        
    }
    public HeterogeneousEnsemble(ArrayList<Classifier> cl){
        ensemble=new ArrayList<Classifier>(cl);
        weights=new double[ensemble.size()];
        w=WeightType.EQUAL;
    }
    public HeterogeneousEnsemble(Classifier[] cl){
        ensemble=new ArrayList<Classifier>();
        for(Classifier c:cl)
            ensemble.add(c);
        weights=new double[ensemble.size()];
        w=WeightType.EQUAL;
    }
    public void useCVWeighting(boolean c){
        if(c)
            w=WeightType.CV;
        else
            w=WeightType.EQUAL;
    }
    @Override
    public void buildClassifier(Instances data) throws Exception {
        train=data;
        for(Classifier c:ensemble)
            c.buildClassifier(train);
//Weighting for voting here    
        switch(w){
            case EQUAL:
                for(int i=0;i<ensemble.size();i++)
                    weights[i]=1.0/ensemble.size();
                break;
            case CV:
                findCVWeights();
                break;
            default:
                System.out.println("Error: weight method not implemented");
                throw new UnsupportedOperationException();
        }
    }
    @Override
    public double[] distributionForInstance(Instance ins){
        double[] dist,temp;
        dist=new double[ins.numClasses()];
        for(int i=0;i<ensemble.size();i++){
            try{
                Classifier c=ensemble.get(i);
                temp=c.distributionForInstance(ins);
                for(int j=0;j<dist.length;j++)
                    dist[j]+=weights[i]*temp[j];
                
                
            }catch(Exception e){
                e.printStackTrace();
                System.out.println("Error classifying instance with classifier ");
                System.exit(0);
            }
        }
        double x=dist[0];
        for(int i=1;i<dist.length;i++)
            x+=dist[i];
        for(int i=0;i<dist.length;i++)
            dist[i]/=x;
        return dist;
    }
	private static final double THRESHOLD1=100;
    public void findCVWeights() throws Exception {
            weights=new double[ensemble.size()];
            double sum=0,sumSq=0;
            int folds=train.numInstances();
            if(folds>THRESHOLD1){
                            folds=10;
            }
            System.out.print("\n Finding CV Accuracy WITHIN ensemble: ");
            for(int i=0;i<ensemble.size();i++){
                Evaluation evaluation = new Evaluation(train);
                     evaluation.crossValidateModel(ensemble.get(i), train, folds, new Random());
                     weights[i]=1-evaluation.errorRate();
                     sum+=weights[i];
                     sumSq+=weights[i]*weights[i];
                     System.out.print(","+weights[i]);
            }
            System.out.print("\n");
            for(int i=0;i<weights.length;i++)
                weights[i]/=sum;
            weightsSD=(sumSq - sum*sum/(weights.length))/(weights.length-1);
            weightsSD=Math.sqrt(weightsSD);
    }

    @Override
    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    public double[] getWeights(){ return weights;}
    public double getWeightsSD(){ return weightsSD;}
    
    
}
