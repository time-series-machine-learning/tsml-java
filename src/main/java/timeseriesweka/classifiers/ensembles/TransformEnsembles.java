package timeseriesweka.classifiers.ensembles;
import timeseriesweka.filters.PowerSpectrum;
import timeseriesweka.filters.ACF;
import java.util.ArrayList;
import java.util.Random;
import weka.attributeSelection.PrincipalComponents;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import timeseriesweka.classifiers.FastDTW_1NN;
import weka.classifiers.lazy.kNN;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.NormalizeAttribute;
import weka.filters.SimpleBatchFilter;

public class TransformEnsembles extends AbstractClassifier{
    enum TransformType {TIME,PS,ACF}; 
        SimpleBatchFilter transform;
	Classifier[] classifiers;
        
    
	boolean useWeights=false;
	boolean normaliseAtts=false;
	ArrayList<Instances> train=new ArrayList<Instances>();
	double[] transformWeights;
	double[] cvWeights;	//Store these once only.
	int numInstances;
	Classifier[] all;
	Classifier base=new kNN(1);	//Default Base classifier
	Classifier baseTime=new FastDTW_1NN();	//Default Base classifier for time domain
        
	PowerSpectrum ps;
	ACF acf;
	PrincipalComponents pca;
	NormalizeAttribute nPs,nAcf,nPca;
	ArrayList<double[][]> predictions=new ArrayList<double[][]>();
	static double CRITICAL=2.32;
	int testPos=0;
	
	int nosTransforms=4;
	double weightPower=1;
	boolean rebuild=true;
	public TransformEnsembles(){
            super();
        }
	public void setWeightPower(double x){weightPower=x;}
	public void rebuildClassifier(boolean x){rebuild =x;}
	public enum WeightType{EQUAL,CV,BEST,STEP}
	
	private WeightType w=WeightType.EQUAL;
	public void setWeightType(int x){
		switch(x){
		case 0:
			w=WeightType.EQUAL;
			break;
		case 1:
			w=WeightType.CV;
			break;
		case 2:
			w=WeightType.BEST;
			break;
		case 3:
			w=WeightType.STEP;
			break;
		}
	}
	public void setWeightType(WeightType x){
		w=x;
	}
	public void setBaseClassifier(Classifier c){
		base=c;
	}
	
	
	private static final double THRESHOLD1=100;
	private static final double THRESHOLD2=1000;
	
	private void init(Instances data){
		numInstances=data.numInstances();
		base=new kNN(1);	//Default Base classifier
		train=new ArrayList<Instances>();
		ps=new PowerSpectrum();
		acf=new ACF();
		acf.setMaxLag((int)(data.numAttributes()-data.numAttributes()*.1));
		pca=new PrincipalComponents (); 
		predictions=new ArrayList<double[][]>();
	
	}
	public void findWeights() throws Exception{
		
		transformWeights=new double[nosTransforms];
		testPos=0;
		switch(w){
		case EQUAL:
			for(int i=0;i<nosTransforms;i++)
				transformWeights[i]=1.0/nosTransforms;
			break;
		case BEST:
			if(cvWeights==null)
				findCVWeights();
			//Set max to 1, rest to zero. If zero, data type will not be built in buildClassifier or used in distributionForInstance
			int max=0;
			for(int i=1;i<cvWeights.length;i++){
				if(cvWeights[i]>cvWeights[max])
					max=i;
			}
			for(int i=0;i<transformWeights.length;i++){
				if(i==max)
					transformWeights[i]=1.0;
				else
					transformWeights[i]=0.0;
			}		
			System.out.print("Best Weight is ");
			switch(max){
			case 0:
				System.out.println("TIME");
				break;
			case 1:
				System.out.println("POWERSPECTRUM");
				break;
			case 2:
				System.out.println("ACF");
				break;
			case 3:
				System.out.println("PCA");
				break;
			}
			break;
		case CV:
			System.out.println("CV Weights");
			if(cvWeights==null)
				findCVWeights();
	//Set transform weights with CV
			double sum=0;
			for(int i=0;i<cvWeights.length;i++){
				sum+=Math.pow(cvWeights[i],weightPower);
			}
			for(int i=0;i<cvWeights.length;i++)
				transformWeights[i]=Math.pow(cvWeights[i],weightPower)/sum;				
			break;
		case STEP:
			System.out.println("STEP Weights");
			if(cvWeights==null)
				findCVWeights();
			//Find the difference between each accuracy, ignore those significantly worse than the best at 10% level
			//Find the most accurate
			max=0;
			for(int i=1;i<cvWeights.length;i++){
				if(cvWeights[i]>cvWeights[max])
					max=i;
			}
			
			//2.1 Work out critical region for alpha
			int n=numInstances;
			double p=cvWeights[max], q=(1-p);
			double sd=p*q;
			sd/=n;
			sd=Math.sqrt(sd);
			for(int j=0;j<cvWeights.length;j++){
				if(j==max)
					transformWeights[j]=Math.pow(cvWeights[j],weightPower);
				else{
					double z=(cvWeights[max]-cvWeights[j])/sd;
					System.out.println(" Max trans ="+max+" z value for "+j+" = "+z);
					if(z<CRITICAL)	//Cant reject H0, keep this transform
						transformWeights[j]=Math.pow(cvWeights[j],weightPower);
					else	//Reject this one
						transformWeights[j]=0;
				}
			}
			//Normalise		
			sum=transformWeights[0];
			for(int i=1;i<transformWeights.length;i++)
				sum+=transformWeights[i];
			for(int i=0;i<transformWeights.length;i++)
				transformWeights[i]/=sum;
			break;
		}

	}
	
	public void buildClassifier(Instances data) throws Exception {
//Sometimes I just want to re-weight it, which must be done with findWeights(). 
//		rebuild stays true by default unless explicitly set by rebuildClassifier(boolean f)
// this is just a bit of a hack to speed up experiments,
		if(rebuild){	
			System.out.println("Build whole ...");
			init(data); //Assume its already standardised
			train.add(data);
			Instances t1=ps.process(data);
			Instances t2=acf.process(data);
			if(normaliseAtts){
				nPs=new NormalizeAttribute(t1);
				t1=nPs.process(t1);
				nAcf=new NormalizeAttribute(t2);
				t2=nAcf.process(t2);
			}
			pca.buildEvaluator(data);
			Instances t3=pca.transformedData(data);
			train.add(t1); //
			train.add(t2);
			train.add(t3);
			nosTransforms=train.size();
			findWeights();
			all= AbstractClassifier.makeCopies(base,train.size());
                        all[0]=AbstractClassifier.makeCopy(baseTime);
			for(int i=0;i<all.length;i++){
				all[i].buildClassifier(train.get(i));
			}
		}
	}
	
	
	public double[] distributionForInstance(Instance ins) throws Exception{
		double[][] preds;
		if(rebuild){	
			preds=new double[nosTransforms][];
			if(all[0]!=null)
				preds[0]=all[0].distributionForInstance(ins);
	//Nasty hack because I've implemented them as batch filters		
			Instances temp=new Instances(train.get(0),0);
			temp.add(ins);
			Instances temp2;
			if(all[1]!=null){
				temp2=ps.process(temp);
				if(normaliseAtts){
					temp2=nPs.process(temp2);
				}
				preds[1]=all[1].distributionForInstance(temp2.instance(0));
			}
			if(all[2]!=null){
				temp2=acf.process(temp);
				if(normaliseAtts){
					temp2=nAcf.process(temp2);
				}
				preds[2]=all[2].distributionForInstance(temp2.instance(0));
			}
			if(all[3]!=null){
				Instance t= pca.convertInstance(ins);
				preds[3]=all[3].distributionForInstance(t);
			}
			predictions.add(preds);
		}
		else{
			preds=predictions.get(testPos);
			testPos++;
		}
		//Weight each
		double[] dist=new double[ins.numClasses()];
		for(int i=0;i<nosTransforms;i++){
			if(transformWeights[i]>0){	//Equivalent to all[i]!=null
				for(int j=0;j<dist.length;j++)
					dist[j]+=transformWeights[i]*preds[i][j];
			}
		}
		
		return dist;
	}

//This ALWAYS recalculates the CV accuracy	
	public void findCVWeights() throws Exception {
		cvWeights=new double[nosTransforms];
		int folds=numInstances;
		if(folds>THRESHOLD1){
				folds=10;
		}
		System.out.print("\n Finding CV Accuracy: ");
		for(int i=0;i<nosTransforms;i++){
			 Evaluation evaluation = new Evaluation(train.get(i));
                         if(i==0)
                            evaluation.crossValidateModel(AbstractClassifier.makeCopy(baseTime), train.get(i), folds, new Random());
                         else
                             evaluation.crossValidateModel(AbstractClassifier.makeCopy(base), train.get(i), folds, new Random());
			 cvWeights[i]=1-evaluation.errorRate();
			 System.out.print(","+cvWeights[i]);
		}
		 System.out.print("\n");
	}
	
	public String getWeights(){ 
		String str="";
		for(int i=0;i<transformWeights.length;i++)
			str+=transformWeights[i]+",";
		return str;
	}
	public String getCV(){ 
		String str="";
		for(int i=0;i<cvWeights.length;i++)
			str+=cvWeights[i]+",";
		return str;
	}
	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}
	public static void main(String[] args){
//Load up Beefand test only on that
		
		
	}
	
}
