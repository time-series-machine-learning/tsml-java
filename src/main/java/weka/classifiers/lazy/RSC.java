package weka.classifiers.lazy;

import java.util.ArrayList;
import java.util.Random;
import java.util.TreeSet;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.*;

/**
 *
 * @author Aaron. Implementation 
 */
public class RSC extends AbstractClassifier implements Randomizable{
    private int alpha;
    private NormalizableDistance distanceFunc;
    private TreeSet<Instance> uncoveredCases;
    private Instances allCases;
    private ArrayList<Instance> T;
    private ArrayList<Sphere> sphereSet;
    private int randSeed=100;
    private Random random = new Random(randSeed);
    

    private boolean crossValidateAlpha=false;
    
    public RSC() {
        crossValidate(true);
        distanceFunc = new EuclideanDistance();
    }
    public RSC(int a) {
        this.alpha = a;
        distanceFunc = new EuclideanDistance();
    }
    public final void crossValidate(boolean b){
        crossValidateAlpha=b;
    }
   @Override
    public void setSeed(int seed) {
        random.setSeed(seed);
        randSeed=seed;
    }

    @Override
    public int getSeed() {
        return randSeed;
    }

    //default distance function is Euclidean
    @Override
    public void buildClassifier(Instances inst){
        
        if(crossValidateAlpha){
//This is a REALLY inefficient way to do this cross validation, it is just a first go
// Spheres are recalculated for every single fold!            
            double bestAccuracy=0;
            int maxAlpha=inst.numInstances()/10;
            RSC r;
            int folds=10;
            for(int a=1;a<maxAlpha;a++){
//Eval                
               r=new RSC(1);
               try{
                    Evaluation e=new Evaluation(inst);
                    e.crossValidateModel(r, inst, folds, random);
                    double acc=e.correct()/inst.numInstances();
                    if(acc>bestAccuracy){
                        bestAccuracy=acc;
                        this.alpha=a;
                    }
               }catch(Exception e){
                   e.printStackTrace();
                   System.exit(0);
               }
            }
            
            
        }
        
        
        sphereSet = new ArrayList();
        
        uncoveredCases = new TreeSet<Instance>(new InstanceComparator());
        distanceFunc.setInstances(inst);
        allCases = inst;
//        uncoveredCases.addAll(i);
        for(int j=0;j<inst.numInstances();j++)
            uncoveredCases.add(inst.instance(j));

        
        //add members of allCases to covered as their covered until allCases is empty.
        while(uncoveredCases.size()>0){
            
            //randomly pick an instance
            int rand = (int)(random.nextDouble()*uncoveredCases.size());
            
            Instance[] tempArray = new Instance[uncoveredCases.size()];
            uncoveredCases.toArray(tempArray);
            Instance temp =  tempArray[rand];
            uncoveredCases.remove(temp);

            //find closest instance that is not the same class value.
            Instance edge = null;
            double distance = Double.MAX_VALUE;
            for(int j=0; j<allCases.numInstances();j++){
               Instance temp2 =allCases.instance(j);
               double tempDist = distanceFunc.distance(temp,temp2);
               //if its in the sphere and isn't the same class.
               if((tempDist <= distance) && (temp.classValue() != temp2.classValue())){
                   distance = tempDist;
                   edge = temp2;
               }
            }

            Sphere TempSphere = new Sphere(temp,distance);

            //find the instances that are covered by the sphere.
            //i feel i could do some optimization here because there ordered?
            //but there ordered with respect to each other and does that mean they'll
            //be close togerger. Who knows?
            //if(uncoveredCases.size()>0){
            T= new ArrayList();
            T.add(edge);
            //find all cases that are inside the sphere.
            for(int j=0; j<allCases.numInstances();j++){
                Instance tempInst = allCases.instance(j);
                double tempDist = distanceFunc.distance(temp,tempInst);
                //if its in the sphere and isn't itself.
                if((tempDist <= distance) && (tempDist != 0)){
                    T.add(tempInst);
                }  
            }


            //check the number of instances covered.
            if(T.size()>=alpha){
                for(int j=0;j<T.size();j++){
                    //remove from uncovered
                    Instance temp1 =T.get(j);
                    uncoveredCases.remove(temp1);
                }
                sphereSet.add(TempSphere);
            }
           //}
        }
    }


    //returns the instances classValue if its inside its sphere. Else it retursn the closest sphere edge.
    @Override
    public double classifyInstance(Instance i) throws Exception{
        int closestSphere =0;
        int closestCentre=-1;
        double previousDistance = Double.MAX_VALUE;
        if(sphereSet.size() > 0){
            for(int j=0;j<sphereSet.size();j++){
                Sphere temp = sphereSet.get(j);
                double distance = distanceFunc.distance(temp.getCentre(),i);
                //if its inside the sphere
                if(distance <= temp.getRadius()){
                    if(closestCentre!=-1){
                       if(distance < distanceFunc.distance(sphereSet.get(closestCentre).getCentre(),i))
                            closestCentre=j;
                    }
                    else
                        closestCentre =j;
                    //return sphereSet.get(j).getCentre().classValue();
                }
                else if(distance-temp.getRadius() <= previousDistance){
                    previousDistance = distance-temp.getRadius();
                    closestSphere = j;
                }
                //if its not, then check which sphere edge is closest.
            }
            if(closestCentre!=-1)
                return sphereSet.get(closestCentre).getCentre().classValue();
            else
                return sphereSet.get(closestSphere).getCentre().classValue();
        }
        else
            throw new Exception("No Spheres in the set");
        
    }

    public void setDistanceFunc(NormalizableDistance in){
        distanceFunc =in;
    }

    public ArrayList<Sphere> getSphereSet(){
        return sphereSet;
    }

    @Override
    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

 
    public static class Sphere {
        private Instance centre;
        private double radius;

        public Sphere(Instance c, double r){
            this.centre =c;
            this.radius =r;
        }

        public Instance getCentre(){
            return centre;
        }

        public double getRadius(){
            return radius;
        }

    }
    public static void main(String[] args){
        System.out.println(" Test harness not implemented");
        
            }
   
}
