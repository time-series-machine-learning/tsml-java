/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.examples;

import weka.core.Instances;
import weka.filters.Filter;
import timeseriesweka.filters.ACF;
import timeseriesweka.filters.PowerSpectrum;

/**
 *
 * @author ajb
 */
public class TransformExamples {
    public static Instances acfTransform(Instances data){
        ACF acf=new ACF();
	acf.setMaxLag(data.numAttributes()/4);
        Instances acfTrans=null;
        try{
            acf.setInputFormat(data);
            acfTrans=Filter.useFilter(data, acf);
        }catch(Exception e){
                System.out.println(" Exception in ACF harness="+e);
		e.printStackTrace();
               System.exit(0);
        }
      
            return acfTrans;
    }
    public static Instances psTransform(Instances data){
        PowerSpectrum ps=new PowerSpectrum();
        Instances psTrans=null;
        try{
            ps.setInputFormat(data);
            psTrans=Filter.useFilter(data, ps);
            ps.truncate(psTrans, data.numAttributes()/4);
        }catch(Exception e){
                System.out.println(" Exception in ACF harness="+e);
		e.printStackTrace();
               System.exit(0);
        }
           return psTrans;
    }
    
    
}
