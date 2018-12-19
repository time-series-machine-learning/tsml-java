/*
     * copyright: Anthony Bagnall
 * 
 * */
package timeseriesweka.filters;

import development.DataSets;
import fileIO.OutFile;
import java.io.FileReader;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import timeseriesweka.classifiers.FastDTW_1NN;
import weka.classifiers.lazy.kNN;
import weka.core.*;
import weka.filters.SimpleBatchFilter;

public class Cosine extends SimpleBatchFilter {
	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
		Attribute a;
		FastVector fv=new FastVector();
		FastVector atts=new FastVector();

		
		  for(int i=0;i<inputFormat.numAttributes()-1;i++)
		  {
//Add to attribute list                          
                        String name = "Cosine_"+i;
                        atts.addElement(new Attribute(name));
		  }
                    //Get the class values as a fast vector			
                    Attribute target =inputFormat.attribute(inputFormat.classIndex());

                    FastVector vals=new FastVector(target.numValues());
                    for(int i=0;i<target.numValues();i++)
                            vals.addElement(target.value(i));
                    atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
                Instances result = new Instances("COSINE"+inputFormat.relationName(),atts,inputFormat.numInstances());
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
                        double c=k*(i+0.5)*(Math.PI/n);
                        fk+=oldInst.value(i)*Math.cos(c);
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
            String s="Beef";
            OutFile of1 = new OutFile("C:\\Users\\ajb\\Dropbox\\test\\BeefCosine_TRAIN.arff");
            OutFile of2 = new OutFile("C:\\Users\\ajb\\Dropbox\\test\\BeefCosine_TEST.arff");
            Instances test=utilities.ClassifierTools.loadData(DataSets.dropboxPath+s+"\\"+s+"_TEST");
            Instances train=utilities.ClassifierTools.loadData(DataSets.dropboxPath+s+"\\"+s+"_TRAIN");			
            Cosine cosTransform= new Cosine();
            Sine sinTransform=new Sine();
            Hilbert hilbertTransform= new Hilbert();
            System.out.println(" Data set ="+s);
            try {
                Instances cosTrain=cosTransform.process(train);
                Instances cosTest=cosTransform.process(test);
                of1.writeString(cosTrain+"");
                of2.writeString(cosTest+"");
                System.out.println(" Cosine trans complete");
                FastDTW_1NN a=new FastDTW_1NN();
//                a.normalise(false);
                a.buildClassifier(cosTrain);
                double acc=ClassifierTools.accuracy(cosTest, a);
                System.out.println(" Cosine acc ="+acc);

                
            } catch (Exception ex) {
                  System.out.println("ERROR in Cosine");
            
            }
        
	}
}
