package timeseriesweka.filters;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
/* simple Filter that just creates a new series of differences order k.
 * The new series has k fewer attributes than the original
 * */
public class Differences extends SimpleBatchFilter {
        private int order=1;
        String attName="";
        public void setOrder(int m){order=m;}
	private static final long serialVersionUID = 1L;
        
        public void setAttName(String s){
            attName=s;
        }

        protected Instances determineOutputFormat(Instances inputFormat)
	throws Exception {
	//Check all attributes are real valued, otherwise throw exception
	for(int i=0;i<inputFormat.numAttributes();i++)
		if(inputFormat.classIndex()!=i)
			if(!inputFormat.attribute(i).isNumeric())
				throw new Exception("Non numeric attribute not allowed in Moments");
	//Set up instances size and format. 
	FastVector atts=new FastVector();
	String name;
	for(int i=0;i<inputFormat.numAttributes()-order-1;i++){
		name = attName+"Difference"+order+"_"+(i+1);
		atts.addElement(new Attribute(name));
	}
	if(inputFormat.classIndex()>=0){	//Classification set, set class 
		//Get the class values as a fast vector			
		Attribute target =inputFormat.attribute(inputFormat.classIndex());

		FastVector vals=new FastVector(target.numValues());
		for(int i=0;i<target.numValues();i++)
			vals.addElement(target.value(i));
		atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
	}	
	Instances result = new Instances("Difference"+order+inputFormat.relationName(),atts,inputFormat.numInstances());
	if(inputFormat.classIndex()>=0){
		result.setClassIndex(result.numAttributes()-1);
	}
	return result;
}

	
@Override
public String globalInfo() {

return null;
}

@Override
public Instances process(Instances inst) throws Exception {
	Instances output=determineOutputFormat(inst);
	

	for(int i=0;i<inst.numInstances();i++){
	//1. Get series: 
		double[] d=inst.instance(i).toDoubleArray();
	//2. Remove target class
		double[] temp;
		int c=inst.classIndex();
		if(c>=0)
                {
			temp=new double[d.length-1];
                        System.arraycopy(d,0,temp,0,c);
 //                       if(c<temp.length)
 //                           System.arraycopy(d,c+1,temp,c,d.length-(c+1));
			d=temp;
		}
        //3. Create Difference series        
		double[] diffs;
                if(c>=0)
                    diffs=new double[output.numAttributes()-1];
                else
                    diffs=new double[output.numAttributes()];
                for(int j=0;j<diffs.length;j++)
                    diffs[j]=d[j]-d[j+order];

          //Extract out the terms and set the attributes
            Instance newInst=null;
            if(c>=0)
                    newInst=new DenseInstance(diffs.length+1);
            else
                    newInst=new DenseInstance(diffs.length);

            for(int j=0;j<diffs.length;j++){
                        newInst.setValue(j,diffs[j]);
            }
            if(c>=0)
                    newInst.setValue(output.classIndex(), inst.instance(i).classValue());
            output.add(newInst);     
        }	
	return output;
}

	
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}

	public static void main(String[] args) {
/**Debug code to test SummaryStats generation: 
	
		
            try{
                Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\Beef\\Beef_TRAIN");
//                Instances filter=new SummaryStats().process(test);
               SummaryStats m=new SummaryStats();
               m.setInputFormat(test);
               Instances filter=Filter.useFilter(test,m);
               System.out.println(filter);
            }
            catch(Exception e){
               System.out.println("Exception thrown ="+e);
               e.printStackTrace();
               
            }

*/
        }
}
