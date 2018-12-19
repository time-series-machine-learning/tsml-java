package weka.classifiers.lazy;

import java.util.Arrays;

import weka.attributeSelection.*;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;


/* 23/7/11: ajb
 * This class bridges the two weka components required to select a subset of attributes
 * 
 * 1. weka.attributeSelection.ASEvaluation
 Abstract base class for the subtypes 
 * 	AttributeEvaluator: evaluates attributes individually: Concrete subtypes:
 * 		ChiSquaredAttributeEval, GainRatioAttributeEval, InfoGainAttributeEval, 
 * 		OneRAttributeEval, ReliefFAttributeEval, SVMAttributeEval, 
 * 		SymmetricalUncertAttributeEval, UnsupervisedAttributeEvaluator
 *  SubsetEvaluator: evaluate a subset
 *  	CfsSubsetEval, ConsistencySubsetEval, HoldOutSubsetEvaluator, 
 *  	UnsupervisedSubsetEvaluator, WrapperSubsetEval
 * 
 * 2. weka.attributeSelection.ASSearch
 Uses the ASEvaluation to choose a subset of attributes
 BestFirst, ExhaustiveSearch, GeneticSearch, GreedyStepwise, RaceSearch, RandomSearch, Ranker, RankSearch

23/7/11: Currently throws an Exception if you combine a single attribute ranker (AttributeEvaluator subclass) with a subset search technique (all of 
ASSearch apart from Ranker. Should deal with this exception, no logic in using anything but Ranker with single attribute evaluator.
 
 */
public class AttributeFilterBridge {
	ASEvaluation eval;
//Note the parameters for the stopping criteria of search vary between
//	implementations, so should be set outside of this class
	int[] attsToKeep;	//The indexes of the original attributes to keep.
	int[] allAtts;	//Sorted array of attributes
	private ASSearch search;
	private Instances data;
	protected double prop=0.2;
	protected int n=0;
	private boolean useProp=false;
//You have to specify either the data set or the ASEvaluation and ASSearch at creation	
	private AttributeFilterBridge(){}
	
	public AttributeFilterBridge(Instances d){
//Search defaults to 10% of the data set using InformationGain
		data =d;
		eval=new InfoGainAttributeEval();
		Ranker r=new Ranker();
		n=(int)(prop*data.numAttributes());
		if(n==0)
			n++;
//Note this does not seem to work, so we fix it by just selecting a subset after generation.		
		r.setNumToSelect(n);
		search=r;
	}
	public AttributeFilterBridge makeCopy(){
		AttributeFilterBridge newAF=new AttributeFilterBridge();
		newAF.search=search;
		newAF.eval=eval;
		return newAF;
	}
	public AttributeFilterBridge(ASEvaluation e,ASSearch s){
		eval=e;
		search=s;
	}
	public void setNosToKeep(int nos){
		useProp=false;
		n=nos;
		if(data!=null) prop=((double)n)/(data.numAttributes()-1);
	}
	public void setProportionToKeep(double p){
		useProp=true;
		prop=p;
		if(data!=null) n=(int)(prop*(data.numAttributes()-1));
	}
	
	public Instances filter(){
		if(data!=null) return filter(data);
		return null;
	}
	public Instances filter(Instances d){
		data=d;
		Instances newD=d;
		int[] atts;
		
		try{
//Build evaluator
			eval.buildEvaluator(d);
//Select attributes
			allAtts=search.search(eval,d);
			
			if(useProp)
				n=(int)(prop*(d.numAttributes()-1));
			if(n==0) n++;
			atts=new int[n];
			if(n<allAtts.length)
				System.arraycopy(allAtts, 0, atts, 0, n);
			else
				atts=allAtts;
			//Sort
			Arrays.sort(atts);
//Create clone data set, then remove attributes
			newD=new Instances(d);
			int nosDeleted=0;
			int nosKept=0;
			int dataPos=0;
				//Advance to the next to keep
			while(dataPos<newD.numAttributes()-1 && nosKept<atts.length){
				while(dataPos!=atts[nosKept]-nosDeleted && dataPos<newD.numAttributes()-1){
					newD.deleteAttributeAt(dataPos);
					nosDeleted++;
				}
				nosKept++;
				dataPos++;
			}
			while(dataPos<newD.numAttributes()-1)
				newD.deleteAttributeAt(dataPos);
			attsToKeep=atts;

		}catch(Exception e){
			System.out.println("Exception thrown in AttributeFilterBridge ="+e);
			e.printStackTrace();
			System.exit(0);
		}
		return newD;
	}
	
        
	public Instance filterInstance(Instance ins){
		int nosDeleted=0;
		int nosKept=0;
		int dataPos=0;
		Instance newIns=new DenseInstance(ins);
			//Advance to the next to keep
		while(dataPos<newIns.numAttributes()-1 && nosKept<attsToKeep.length){
			while(dataPos!=attsToKeep[nosKept]-nosDeleted && dataPos<newIns.numAttributes()-1){
				newIns.deleteAttributeAt(dataPos);
				nosDeleted++;
			}
			nosKept++;
			dataPos++;
		}
		while(dataPos<newIns.numAttributes()-1)
			newIns.deleteAttributeAt(dataPos);
		return newIns;
		
	}
	public String toString(){
		String str="\n Attributes retained =";
		for(int i=0;i<attsToKeep.length;i++)
			str+=" "+attsToKeep[i];
		return str;
	}
        
/** So this below is to generate different sets from the same ranking. 
	Usage
	AttributeFilterBridge af=new AttributeFilterBridge();
	//Set eval and search if required
	af.rankAttributes(Instances data);
	double prop=0.5; //Proportion of attributes to keep
	Instances fTrain ]= af.filterBest(prop);	//Will not work out the ranks again
	
**/	
	public void rankAttributes(Instances d){
		data=d;
		try{
//Build evaluator
			eval.buildEvaluator(d);
//Select attributes
			allAtts=search.search(eval,d);
//Sort
			Arrays.sort(allAtts);
		}catch(Exception e){
			e.printStackTrace();
			System.out.println(" Exception in trank atts");
			System.exit(0);
		}
	}
/*	public Instances rankAttributes(double p){
		Instances newD=new Instances(data);
		prop=p;
		int[] atts;
		try{
			if(useProp)
				n=(int)(prop*(data.numAttributes()-1));
			if(n==0) n++;
			atts=new int[n];
			System.arraycopy(allAtts, 0, atts, 0, n);
//Create clone data set, then remove attributes
			newD=new Instances(d);
			int nosDeleted=0;
			int nosKept=0;
			int dataPos=0;
				//Advance to the next to keep
			while(dataPos<newD.numAttributes()-1 && nosKept<atts.length){
				while(dataPos!=atts[nosKept]-nosDeleted && dataPos<newD.numAttributes()-1){
					newD.deleteAttributeAt(dataPos);
					nosDeleted++;
				}
				nosKept++;
				dataPos++;
			}
			while(dataPos<newD.numAttributes()-1)
				newD.deleteAttributeAt(dataPos);
			attsToKeep=atts;

	
	}
*/	
        
/*This will test the information gain scores and that the correct attributes        
 * are retained
 */
        public static void testCorrectness(){
            
            
        }
        
        
        public static void main(String[] args){
/** To check
		1. The scoring is correct: Scoring is performed by the evaluator, so is not suitable for testing here
		2. That the number retained and proportion retained works robustly
		3. That the rank order list is sorted correctly 
		4. That the correct attributes are retained.
		*/
/*		String path="C:\\Research\\Data\\WekaTest\\";
		
		Instances beef=utilities.ClassifierTools.loadData(path+"Beef_TRAIN");
//Iris has 4 attributes, checking that 
		Instances data=utilities.ClassifierTools.loadData(path+"irisSmall");
		AttributeFilterBridge af=new AttributeFilterBridge(data);
		Instances d2=af.filter();
/* 1. Check that the ranking is correct. Iris has four attributes and three class values. Not sure what the Chi-Sq filter does, 
 * the info gain will look at all splits. Leave this.	
*/
/*		2. That the number retained and proportion retained works robustly

//Beef has 470 attributes. 10% should retain 47, 1% keep 4 (?), 25% keep 117 		
		AttributeFilterBridge beefFilter=new AttributeFilterBridge(beef);
		beefFilter.setProportionToKeep(0.1);
		Instances b2=beefFilter.filter();
		System.out.println("10% number of atts = "+(b2.numAttributes()-1));
		beefFilter.setProportionToKeep(0.01);
		b2=beefFilter.filter();
		System.out.println("1% number of atts = "+(b2.numAttributes()-1));
		beefFilter.setProportionToKeep(0.25);
		b2=beefFilter.filter();
		System.out.println("25% number of atts = "+(b2.numAttributes()-1));
		beefFilter.setNosToKeep(10);
		b2=beefFilter.filter();
		System.out.println("10 atts = "+(b2.numAttributes()-1));
		beefFilter.setNosToKeep(100);
		b2=beefFilter.filter();
		System.out.println("100 number of atts = "+(b2.numAttributes()-1));
		*/
		
/* 2. Check the right attributes are removed
 		2.1 Print out the ranker list for all
 		2.2 Check against reduced sorted list

		AttributeFilterBridge beefFilter=new AttributeFilterBridge(beef);
		beefFilter.setProportionToKeep(0.05);
		Instances b2=beefFilter.filter();
		System.out.println("5% number of atts = "+(b2.numAttributes()-1));
	
 */		
		/*

		System.out.println("Attribute filter \t"+af);
			
		System.out.println("New data ="+d2);

		
		System.out.println(" Number of attributes in new data ="+d2.numAttributes());
//		System.out.println(d2);
/*		ASEvaluation e=new ChiSquaredAttributeEval();
		ASSearch s=new Ranker();
		AttributeFilterBridge af2=new AttributeFilterBridge(e,s);
		Instances d3=af2.filter(data);
		System.out.println(" Number of attributes in new data ="+d3.numAttributes());
	*/	
		
//2. Check the correct attributes are being removed
		

	}
}
