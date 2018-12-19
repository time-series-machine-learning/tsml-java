/*
 * This class contains a static method for generating parameterised simulated
 * time-series datasets. The datasets are designed for shapelet approaches.
 * This will produce a two-class problem.
 */
package statistics.simulators;

import fileIO.OutFile;
import java.util.Random;
import weka.core.Instances;
import statistics.simulators.ShapeletModel.ShapeType;
/**
 *
 * @author Jon Hills
 * j.hills@uea.ac.uk
 */
public class SimulateShapeletData extends DataSimulator{
       
    
    /**
     * This method creates and returns a set of Instances representing a
     * simulated two-class time-series problem.
     * 

     * @param seriesLength The length of the series. All time series in the
     * dataset are the same length.
     * @param casesPerClass An array of two integers indicating the number of 
     * instances of class 0 and class 1.
     * @return Instances representing the time-series dataset. The Instances
     * returned will be empty if the casesPerClass parameter does not contain
     * exactly two values.
     */
    public static Instances generateShapeletData(int seriesLength, int []casesPerClass)
    {
        
        if( casesPerClass.length != 2)
        {
            System.err.println("ONLY WORKS WITH TWO CLASS PROBS AT THE MOMENT");
            int[] tmp = {0,0};
            casesPerClass = tmp;
            
        }
        ShapeletModel[] shapeMod = new ShapeletModel[casesPerClass.length];
        populateShapeletArray(shapeMod, seriesLength);
        DataSimulator sim = new DataSimulator(shapeMod);
        sim.setSeriesLength(seriesLength);
        sim.setCasesPerClass(casesPerClass);
        Instances d=sim.generateDataSet();
        return d;          
    }
    
    /**
     * This is a support method for generateShapeletData
     * 
     * @param array An array of two ShapeletModel2 models, representing the 
     * simulated shapes inserted into the respective classes.
     * @param seriesLength The length of the series.
     */
    private static void populateShapeletArray(ShapeletModel [] s, int seriesLength)
    {
        double[] p1={seriesLength,1};
        double[] p2={seriesLength,1};
       
//Create two ShapeleModels with different base Shapelets        
        s[0]=new ShapeletModel(p1);        
        
        ShapeletModel.ShapeType st=s[0].getShapeType();
        s[1]=new ShapeletModel(p2);
        while(st==s[1].getShapeType()){ //Force them to be different types of shape
            s[1]=new ShapeletModel(p2);
        }
    }
    
    public static void checkGlobalSeedForIntervals(){
        Model.setDefaultSigma(0);
        Model.setGlobalRandomSeed(0);
        Instances d=generateShapeletData(100,new int[]{2,2});
        OutFile of = new OutFile("C:\\Temp\\randZeroNoiseSeed0.csv");
       of.writeLine(d.toString());
        Model.setDefaultSigma(0.1);
        Model.setGlobalRandomSeed(1);
        System.out.println(" NOISE 0");
         d=generateShapeletData(100,new int[]{2,2});
        of = new OutFile("C:\\Temp\\randUnitNoiseSeed1.csv");
       of.writeLine(d.toString());
        Model.setDefaultSigma(0);
        Model.setGlobalRandomSeed(0);
        System.out.println(" NO NOISE 1");
         d=generateShapeletData(100,new int[]{2,2});
        of = new OutFile("C:\\Temp\\randZeroNoiseSeed0REP.csv");
       of.writeLine(d.toString());
        Model.setDefaultSigma(0.1);
        Model.setGlobalRandomSeed(1);
        System.out.println(" NOISE 1");
         d=generateShapeletData(100,new int[]{2,2});
        of = new OutFile("C:\\Temp\\randUnitNoiseSeed1REP.csv");
       of.writeLine(d.toString());
 }
    

    public static void main(String[] args)
    {
        
        checkGlobalSeedForIntervals();
        System.exit(0);
        int[] casesPerClass = {5,5};
        int seriesLength = 100;
        Model.setDefaultSigma(0);
        Model.setGlobalRandomSeed(0);
        System.out.println("Model seed ="+Model.getRandomSeed());
        Instances data = SimulateShapeletData.generateShapeletData(seriesLength,casesPerClass);
        System.out.println("DATA "+data);
        System.out.println("Model seed AFTER ="+Model.getRandomSeed());
        OutFile out=new OutFile("C:\\temp\\ShapeletData.csv");
        out.writeString(data.toString());
    }
      
}
