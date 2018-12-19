/**
 * This is the discord model.
 * 
 * All data hasa big spike, class 1 has a small recurrence.
 */
package statistics.simulators;
import fileIO.OutFile;
import java.util.*;
import java.io.*;
import statistics.distributions.NormalDistribution;
import statistics.simulators.DictionaryModel.ShapeType;
import static statistics.simulators.Model.rand;
import statistics.simulators.ShapeletModel.Shape;

public class MatrixProfileModelVersion2 extends Model {
    private int nosLocations=2; //
    private int shapeLength=29;
    public static double MINBASE=-2;
    public static double MINAMP=2;
    public static double MAXBASE=2;
    public static double MAXAMP=4;
    DictionaryModel.Shape shape;//Will change for each series
    private static int GLOBALSERIESLENGTH=500;
    private int seriesLength; // Need to set intervals, maybe allow different lengths? 
    private int base=-1;
    private int amplitude=2;
    private int shapeCount=0;
    private boolean invert=false;
    boolean discord=false;
    double[] shapeVals;
    ArrayList<Integer>  locations;
    private static double[] spike1;
    private static double[] spike2;
    
    private static void makeSpikes(){
        spike1=new double[29];
        spike2=new double[29];
        double max=3;
        double min=-2;
        spike1[0]=min;
        spike1[28]=min;
        spike1[15]=max; 
        for(int i=1;i<=14;i++)
           spike1[i]=spike1[i-1]+(max-min)/5;
        for(int i=0;i<29;i++)
           spike2[i]=spike1[i]/5;     
    }
    
    public static int getGlobalLength(){ return GLOBALSERIESLENGTH;}
    public MatrixProfileModelVersion2(){
        shapeCount=0;//rand.nextInt(ShapeType.values().length);
        seriesLength=GLOBALSERIESLENGTH;
        locations=new ArrayList<>();    
        setNonOverlappingIntervals();
        shapeVals=new double[shapeLength];
        generateRandomShapeVals();
    }
    public MatrixProfileModelVersion2(boolean d){
        shapeCount=0;//rand.nextInt(ShapeType.values().length);
        discord=d;
        if(discord)
            nosLocations=1;
        seriesLength=GLOBALSERIESLENGTH;
        locations=new ArrayList<>();    
        setNonOverlappingIntervals();
         shapeVals=new double[shapeLength];
        generateRandomShapeVals();
       
    }
    private void generateRandomShapeVals(){
        for(int i=0;i<shapeLength;i++)
            shapeVals[i]=MINBASE+(MAXBASE-MINBASE)*Model.rand.nextDouble();
    }
    
    public void setSeriesLength(int n){
        seriesLength=n;
    }
    public static void setGlobalSeriesLength(int n){
        GLOBALSERIESLENGTH=n;
    }
   public void setNonOverlappingIntervals(){
//Use Aarons way
       ArrayList<Integer> startPoints=new ArrayList<>();
       for(int i=shapeLength+1;i<seriesLength-shapeLength;i++)
           startPoints.add(i);
        for(int i=0;i<nosLocations;i++){
            int pos=rand.nextInt(startPoints.size());
            int l=startPoints.get(pos);
            locations.add(l);
//+/- windowSize/2
            if(pos<shapeLength/2)
                pos=0;
            else
                pos=pos-shapeLength/2;
            for(int j=0;startPoints.size()>pos && j<(2*shapeLength);j++)
                startPoints.remove(pos);
        }
        Collections.sort(locations);
    }

    @Override
    public void setParameters(double[] p) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    public void setLocations(ArrayList<Integer> l, int length){
        locations=new ArrayList<>(l);    
        shapeLength=length;
    }
    public ArrayList<Integer> getIntervals(){return locations;}
    public int getShapeLength(){ return shapeLength;}
    
    public void generateBaseShape(){
//Randomise BASE and AMPLITUDE
        double b=MINBASE+(MAXBASE-MINBASE)*Model.rand.nextDouble();
        double a=MINAMP+(MAXAMP-MINAMP)*Model.rand.nextDouble();
        ShapeType[] all=ShapeType.values();
            ShapeType st=all[(shapeCount++)%all.length];
            shape=new DictionaryModel.Shape(st,shapeLength,b,a);
//            shape=new DictionaryModel.Shape(DictionaryModel.ShapeType.SPIKE,shapeLength,b,a);
//            System.out.println("Shape is "+shape);
 //        shape.nextShape();
//        shape
    }
    
     @Override   
	public	double[] generateSeries(int n)
	{
           t=0;
           double[] d;
        generateRandomShapeVals();
//Resets the starting locations each time this is called          
           if(invert){
                d= new double[n];
                for(int i=0;i<n;i++)
                   d[i]=-generate();
                invert=false;
           }
           else{
            generateBaseShape();
            d = new double[n];
            for(int i=0;i<n;i++)
               d[i]=generate();
                invert=true;
           }
           return d;
        }
    private double generateConfig1(){
 //Noise
//        System.out.println("Error var ="+error.getVariance());
        double value=0;
//Find the next shape        
        int insertionPoint=0;
        while(insertionPoint<locations.size() && locations.get(insertionPoint)+shapeLength<t)
            insertionPoint++;    
//Bigger than all the start points, set to last
        if(insertionPoint>=locations.size()){ 
            insertionPoint=locations.size()-1;
        }
        int point=locations.get(insertionPoint);
        if(point<=t && point+shapeLength>t)//in shape1
            value=shapeVals[(int)(t-point)];
          else
            value= error.simulate();

            
//            value+=shape.generateWithinShapelet((int)(t-point));
//                System.out.println(" IN SHAPE 1 occurence "+insertionPoint+" Time "+t);

        t++;
        return value;       
    }
    
    private double generateConfig2(){
 //Noise
//        System.out.println("Error var ="+error.getVariance());
        double value=error.simulate();
//Find the next shape        
        int insertionPoint=0;
        while(insertionPoint<locations.size() && locations.get(insertionPoint)+shapeLength<t)
            insertionPoint++;    
//Bigger than all the start points, set to last
        if(insertionPoint>=locations.size()){ 
            insertionPoint=locations.size()-1;
        }
        int point=locations.get(insertionPoint);
        if(insertionPoint>0 && point==t){//New shape, randomise scale
//            double b=shape.getBase()/5;
//            double a=shape.getAmp()/5;
            double b=MINBASE+(MAXBASE-MINBASE)*Model.rand.nextDouble();
           double a=MINAMP+(MAXAMP-MINAMP)*Model.rand.nextDouble();
            shape.setAmp(a);
            shape.setBase(b);
//            System.out.println("changing second shape");
        }
       
        if(point<=t && point+shapeLength>t){//in shape1
            value+=shape.generateWithinShapelet((int)(t-point));
//                System.out.println(" IN SHAPE 1 occurence "+insertionPoint+" Time "+t);
        }
        t++;
        return value;       
    }    
    //Generate point t
    @Override
    public double generate(){
        
        return generateConfig1();
        

    }
    
    public static void generateExampleData(){
        int length=500;
        GLOBALSERIESLENGTH=length;
        Model.setGlobalRandomSeed(3);
        Model.setDefaultSigma(0);
        MatrixProfileModelVersion2 m1=new MatrixProfileModelVersion2();
        MatrixProfileModelVersion2 m2=new MatrixProfileModelVersion2();
        
        double[][] d=new double[20][];
        for(int i=0;i<10;i++){
            d[i]=m1.generateSeries(length);
        }
        for(int i=10;i<20;i++){
            d[i]=m1.generateSeries(length);
        }
        OutFile of=new OutFile("C:\\temp\\MP_ExampleSeries.csv");
        for(int i=0;i<length;i++){
            for(int j=0;j<10;j++)
                of.writeString(d[j][i]+",");
            of.writeString("\n");
        }
        
    }
    public String toString(){
        String str="";
        for(Integer i:locations)
            str+=i+",";
        return str;
    }
    public static void main(String[] args){
        generateExampleData();
        System.exit(0);

//Set up two models with same intervals but different shapes        
        int length=500;
        Model.setGlobalRandomSeed(10);
        Model.setDefaultSigma(0.1);
        MatrixProfileModelVersion2 m1=new MatrixProfileModelVersion2();
        MatrixProfileModelVersion2 m2=new MatrixProfileModelVersion2();
        double[] d1=m1.generateSeries(length);
        double[] d2=m2.generateSeries(length);
        OutFile of=new OutFile("C:\\temp\\MP_Ex.csv");
        for(int i=0;i<length;i++)
            of.writeLine(d1[i]+","+d2[i]);
    }
    
}
