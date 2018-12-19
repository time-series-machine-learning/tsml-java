/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package statistics.simulators;

import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class SimulateWholeSeriesData extends DataSimulator {
        static DataSimulator sim;
   
      public SimulateWholeSeriesData(double[][] paras){
        super(paras);
        for(int i=0;i<nosClasses;i++)
            models.add(new SinusoidalModel(paras[i]));
    }
      public void setWarping(){
          for(Model m:models){
              ((SinusoidalModel)m).setWarp(true);
          }
      }
      
    public static Instances generateWholeSeriesdData(int seriesLength, int []casesPerClass)
    {

        SinusoidalModel[] sin = new SinusoidalModel[casesPerClass.length];
        populateWholeSeriesModels(sin);
        sim = new DataSimulator(sin);
        sim.setSeriesLength(seriesLength);
        sim.setCasesPerClass(casesPerClass);
        Instances d=sim.generateDataSet();
        return d;
        
    }        
//We will use the same sine wave for every class, but just shift the offset
    private static void populateWholeSeriesModels(SinusoidalModel[] m){
//Create two models with same interval but different shape. 
        double[] paras= new double[3];
        //Offet changes per class
        paras[0]=0; //Model.rand.nextDouble();
        paras[1]=1;//Model.rand.nextDouble();
        paras[2]=1;//Model.rand.nextDouble();
        for(int i=0;i<m.length;i++){
            m[i]=new SinusoidalModel(paras);
            paras[0]=1; //Model.rand.nextDouble();
        }       
    }
}
