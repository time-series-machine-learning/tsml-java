package timeseriesweka.filters;


import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
/* simple Filter that just summarises the series
     * copyright: Anthony Bagnall

 * Global stats:
 * 		mean, variance, skewness, kurtosis,min, max, range, 
 * */
public class SummaryStats extends SimpleBatchFilter {
        private int numMoments=4;
        public void setNumMoments(int m){numMoments=4;}
	private static final long serialVersionUID = 1L;

        protected Instances determineOutputFormat(Instances inputFormat)
	throws Exception {
	//Check all attributes are real valued, otherwise throw exception
	for(int i=0;i<inputFormat.numAttributes();i++)
		if(inputFormat.classIndex()!=i)
			if(!inputFormat.attribute(i).isNumeric())
				throw new Exception("Non numeric attribute not allowed in SummaryStats");
	//Set up instances size and format. 
	FastVector atts=new FastVector();
        String source=inputFormat.relationName();
	String name;
	for(int i=0;i<numMoments;i++){
		name =source+"Moment_"+(i+1);
		atts.addElement(new Attribute(name));
	}
	atts.addElement(new Attribute(source+"MIN"));
	atts.addElement(new Attribute(source+"MAX"));
        
	if(inputFormat.classIndex()>=0){	//Classification set, set class 
		//Get the class values as a fast vector			
		Attribute target =inputFormat.attribute(inputFormat.classIndex());

		FastVector vals=new FastVector(target.numValues());
		for(int i=0;i<target.numValues();i++)
			vals.addElement(target.value(i));
		atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
	}	
	Instances result = new Instances("Moments"+inputFormat.relationName(),atts,inputFormat.numInstances());
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
	//For each data, first extract the relevan
	int seriesLength=inst.numAttributes();
	if(inst.classIndex()>=0){
		seriesLength--;
	}
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
                double[] moments=new double[numMoments+2];
/**
 * 
 * 
 * HERE FIND MOMENTS HERE
 * 
 * 
**/
            double max=-Double.MAX_VALUE;
            double min=Double.MAX_VALUE;
            double sum = 0;      
            //Find mean
            for(int j=0;j<d.length;j++){
                sum = sum+d[j];
                if(d[j]>max)
                    max=d[j];
                if(d[j]<min)
                    min=d[j];
            }
            moments[0] = sum/d.length;
            double totalVar=0;
            double totalSkew =0;
            double totalKur =0;
            //Find variance
            for(int j=0;j<d.length;j++) {
                totalVar = totalVar+ (d[j]-moments[0])*(d[j]-moments[0]);
                totalSkew = totalSkew+ (d[j]-moments[0])*(d[j]-moments[0])*(d[j]-moments[0]);
                totalKur = totalKur+ (d[j]-moments[0])*(d[j]-moments[0])*(d[j]-moments[0])*(d[j]-moments[0]);
            }
            
            moments[1] = totalVar/(d.length-1);
            double standardDeviation = Math.sqrt(moments[1]);
            moments[1]=standardDeviation;
            double skew = totalSkew/(standardDeviation*standardDeviation*standardDeviation);
            moments[2] = skew/d.length;
            double kur = totalKur/(standardDeviation*standardDeviation*standardDeviation*standardDeviation);
            moments[3] = kur/d.length;
            
          //Extract out the terms and set the attributes
            Instance newInst=null;
            if(inst.classIndex()>=0)
                    newInst=new DenseInstance(numMoments+2+1);
            else
                    newInst=new DenseInstance(numMoments+2);

            for(int j=0;j<numMoments;j++){
                        newInst.setValue(j,moments[j]);
            }
            newInst.setValue(numMoments,min);
            newInst.setValue(numMoments+1,max);
            if(inst.classIndex()>=0)
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
/**Debug code to test SummaryStats generation: **/
	
		
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
        }

}
