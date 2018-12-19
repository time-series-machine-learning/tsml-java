/*
     * copyright: Anthony Bagnall
 * 
 * */
package timeseriesweka.filters;

import java.io.FileReader;
import java.util.*;
import weka.core.*;
import weka.filters.SimpleBatchFilter;

public class Sine extends SimpleBatchFilter {
	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
		Attribute a;
		FastVector fv=new FastVector();
		FastVector atts=new FastVector();

		
		  for(int i=0;i<inputFormat.numAttributes()-1;i++)
		  {
//Add to attribute list                          
                        String name = "Sine"+i;
                        atts.addElement(new Attribute(name));
		  }
                    //Get the class values as a fast vector			
                    Attribute target =inputFormat.attribute(inputFormat.classIndex());

                    FastVector vals=new FastVector(target.numValues());
                    for(int i=0;i<target.numValues();i++)
                            vals.addElement(target.value(i));
                    atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
                Instances result = new Instances("Sine"+inputFormat.relationName(),atts,inputFormat.numInstances());
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
	public Instances process(Instances instances) throws Exception {
//for k=1 to n: f_k = sum_{i=1}^n f_i cos[(k-1)*(\pi/n)*(i-1/2)] 
//Assumes the class attribute is in the last one for simplicity            
            Instances result = determineOutputFormat(instances);
            Instance newInst,oldInst;            
            int n=instances.numAttributes()-1;
            for(int j=0;j<instances.numInstances();j++) {
                oldInst=instances.instance(j);
                newInst= new DenseInstance(result.numAttributes());
                for(int k=0;k<n;k++){
                  double fk=0;
                    for(int i=0;i<n;i++){
                        double c=(k+1)*(i+1/2)*(Math.PI/n);
                        fk+=oldInst.value(i)*Math.sin(c);
                    }
                    newInst.setValue(k, fk);
                }
                
               newInst.setValue(result.classIndex(), instances.instance(j).classValue());
               result.add(newInst);
            }
            return result;
	}

	public String getRevision() {
		return null;
	}
	public static void main(String[] args){
		Clipping cp=new Clipping();
		Instances data=null;
		String fileName="C:\\Research\\Data\\Time Series Data\\Time Series Classification\\TestData\\TimeSeries_Train.arff";
		try{
			FileReader r;
			r= new FileReader(fileName); 
			data = new Instances(r); 

			data.setClassIndex(data.numAttributes()-1);
			
			System.out.println(" Class type numeric ="+data.attribute(data.numAttributes()-1).isNumeric());
			System.out.println(" Class type nominal ="+data.attribute(data.numAttributes()-1).isNominal());

			Instances newInst=cp.process(data);
			System.out.println(newInst);
		}catch(Exception e)
		{
			System.out.println(" Error ="+e);
			StackTraceElement [] st=e.getStackTrace();
			for(int i=st.length-1;i>=0;i--)
				System.out.println(st[i]);
				
		}
	}
}
