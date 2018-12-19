package weka.classifiers.lazy;

import development.DataSets;
import java.text.DecimalFormat;
import utilities.ClassifierTools;
import weka.core.*;
import weka.core.neighboursearch.NearestNeighbourSearch;

/** Nearest neighbour classifier that extends the weka one but can take 
 * alternative distance functions.
 * @author ajb
 * @version 1.0
 * @since 5/4/09

1. Normalisation: set by method normalise(boolean)
2. Cross Validation: set by method crossValidate(int folds)
3. Use weighting: set by the method weightVotes()

* 
 * */

public class kNN extends IBk {
	protected DistanceFunction dist;
	double[][] distMatrix;
	boolean storeDistance;
	public kNN(){
//Defaults to Euclidean distance 1NN without attribute normalisation		
            super();
            super.setKNN(1);
            EuclideanDistance ed = new EuclideanDistance();
            ed.setDontNormalize(true);
            setDistanceFunction(ed);
	}
	public kNN(int k){
            super(k);
            EuclideanDistance ed = new EuclideanDistance();
            ed.setDontNormalize(true);
            setDistanceFunction(ed);
	}
	public kNN(DistanceFunction df){
            super();
            setDistanceFunction(df);
	}
	
	public final void setDistanceFunction(DistanceFunction df){
		dist=df;
		NearestNeighbourSearch s = super.getNearestNeighbourSearchAlgorithm();
		try{
			s.setDistanceFunction(df);
		}catch(Exception e){
			System.err.println(" Exception thrown setting distance function ="+e+" in "+this);
                        e.printStackTrace();
                        System.exit(0);
		}
	}
//Need to implement the early abandon for the search?	
	public double distance(Instance first, Instance second) {  
		  return dist.distance(first, second);
	  }
//Only use with a Euclidean distance method
	public void normalise(boolean v){
		if(dist instanceof NormalizableDistance)
			((NormalizableDistance)dist).setDontNormalize(!v);
		else
			System.out.println(" Not normalisable");
	}
    @Override
	public void buildClassifier(Instances d){
		Instances d2=d;
		if(filterAttributes){
			d2=filter(d);
		}
		dist.setInstances(d2);
		try{
			super.buildClassifier(d2);
		}catch(Exception e){
			System.out.println("Exception thrown in kNN build Classifier = "+e);
                        e.printStackTrace();
                        System.exit(0);
		}
	}
    @Override
  public double [] distributionForInstance(Instance instance) throws Exception {
	  if(af!=null){
		  Instance newInst=af.filterInstance(instance);
		  return super.distributionForInstance(newInst);
	  }
	  else
		  return super.distributionForInstance(instance);
	
  }	
	public double[] getPredictions(Instances test){
		double[] pred=new double[test.numInstances()];
		try{
			for(int i=0;i<test.numInstances();i++){
				pred[i]=classifyInstance(test.instance(i));
				System.out.println("Pred = "+pred[i]);
			}
		}catch(Exception e){
			System.out.println("Exception thrown in getPredictions in kNN = "+e);
                        e.printStackTrace();
                        System.exit(0);
		}
		return pred;
	}
        public static void test1NNvsIB1(boolean norm){
            System.out.println("FIRST BASIC SANITY TEST FOR THIS WRAPPER");
            System.out.print("Compare 1-NN with IB1, normalisation turned");
            String str=norm?" on":" off";
            System.out.println(str);
            System.out.println("Compare on the UCI data sets");
            System.out.print("If normalisation is off, then there may be differences");
            kNN knn = new kNN(1);
            IBk ib1=new IBk(1);
            knn.normalise(norm);
            int diff=0;
            DecimalFormat df = new DecimalFormat("####.###");
            for(String s:DataSets.uciFileNames){
                Instances train=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-train");
                Instances test=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-test");
                try{
                    knn.buildClassifier(train);
    //                ib1.buildClassifier(train);
                    ib1.buildClassifier(train);
                    double a1=ClassifierTools.accuracy(test, knn);
                    double a2=ClassifierTools.accuracy(test, ib1);
                    if(a1!=a2){
                        diff++;
                        System.out.println(s+": 1-NN ="+df.format(a1)+" ib1="+df.format(a2));
                    }
                }catch(Exception e){
                    System.out.println(" Exception builing a classifier");
                    System.exit(0);
                }
            }
             System.out.println("Total problems ="+DataSets.uciFileNames.length+" different on "+diff);
        }
        
        public static void testkNNvsIBk(boolean norm, boolean crossValidate){
            System.out.println("FIRST BASIC SANITY TEST FOR THIS WRAPPER");
            System.out.print("Compare 1-NN with IB1, normalisation turned");
            String str=norm?" on":" off";
            System.out.println(str);
            System.out.print("Cross validation turned");
            str=crossValidate?" on":" off";
            System.out.println(str);
            System.out.println("Compare on the UCI data sets");
            System.out.print("If normalisation is off, then there may be differences");
            kNN knn = new kNN(100);
            IBk ibk=new IBk(100);
            knn.normalise(norm);
            knn.setCrossValidate(crossValidate);
            ibk.setCrossValidate(crossValidate);
            int diff=0;
            DecimalFormat df = new DecimalFormat("####.###");
            for(String s:DataSets.uciFileNames){
                Instances train=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-train");
                Instances test=ClassifierTools.loadData(DataSets.uciPath+s+"\\"+s+"-test");
                try{
                    knn.buildClassifier(train);
    //                ib1.buildClassifier(train);
                    ibk.buildClassifier(train);
                    double a1=ClassifierTools.accuracy(test, knn);
                    double a2=ClassifierTools.accuracy(test, ibk);
                    if(a1!=a2){
                        diff++;
                        System.out.println(s+": 1-NN ="+df.format(a1)+" ibk="+df.format(a2));
                    }
                }catch(Exception e){
                    System.out.println(" Exception builing a classifier");
                    System.exit(0);
                }
            }
             System.out.println("Total problems ="+DataSets.uciFileNames.length+" different on "+diff);
        }
        
	public static void main(String[] args){
            //test1NNvsIB1(true);		
            //test1NNvsIB1(false);		
          //  testkNNvsIBk(true,false);		
            testkNNvsIBk(true,true);		

	}
        
//FILTER CODE 	
	boolean filterAttributes=false;
	double propAtts=0.5;
	int nosAtts=0;
	AttributeFilterBridge af;
	public void setFilterAttributes(boolean f){ filterAttributes=f;}
//	public void setEvaluator(ASEvaluation a){ eval=a;}
	public void setProportion(double f){propAtts=f;}
	public void setNumber(int n){nosAtts=n;}
	
	private Instances filter(Instances d){
//Search method: Simple rank, evaluating in isolation
		af=new AttributeFilterBridge(d);
		af.setProportionToKeep(propAtts);
		Instances d2=af.filter();
//		Instances d2=new Instances(d);
//Remove all attributes not in the list. Are they sorted??			
		return d2;
	}
        
}
