/*

Interval model for simulators.
Intervals and shapes are fixed for all series from a given class
Shapes differ between models. Shapes are randomised at construction, but and so
the calling class must make sure they are different between classes with calls to
randomiseShape.

Intervals are set externally by this class by calls to generateRandomIntervals 
to generate the intervals for one model and setIntervals for models of other classes


 */
package statistics.simulators;
import fileIO.OutFile;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import statistics.simulators.DictionaryModel.Shape;
import statistics.simulators.DictionaryModel.ShapeType;
import static statistics.simulators.Model.rand;

/**
 *
 * @author ajb
 */
public class IntervalModel extends Model{
    private int nosIntervals=3; //
    private int seriesLength=300; // Need to set intervals, maybe allow different lengths? 
    private int noiseToSignal=4;
    private int intervalLength=seriesLength/(nosIntervals*noiseToSignal);
    private int base=-1;
    private int amplitude=2;
    private Shape baseShape;
        ArrayList<Integer>  locations;
    public IntervalModel(){
        baseShape=new Shape(intervalLength);
        locations=new ArrayList<>();    
    }
    public IntervalModel(int n){
        this();
        nosIntervals=n;
        intervalLength=seriesLength/(nosIntervals*2);
        baseShape.setLength(intervalLength);
        createIntervals();
    }
    public void setNoiseToSignal(int n){
        noiseToSignal=n;
        intervalLength=seriesLength/(nosIntervals*noiseToSignal);
        baseShape.setLength(intervalLength);
    }
    
    public void setNosIntervals(int n){
        nosIntervals=n;
        intervalLength=seriesLength/(nosIntervals*noiseToSignal);
        baseShape.setLength(intervalLength);
    }
    public void setSeriesLength(int n){ 
        seriesLength=n;
        intervalLength=seriesLength/(nosIntervals*noiseToSignal);
        baseShape.setLength(intervalLength);
    }
    public void setBaseShapeType(ShapeType st){
        baseShape.setType(st);
        baseShape.setLength(intervalLength);
//        System.out.println(" Setting base type "+st+" length ="+intervalLength);
    }
    public final void createIntervals(){
        locations=new ArrayList<>(nosIntervals);    
        setNonOverlappingIntervals();
        baseShape.setLength(intervalLength);
    }
    public void setIntervals(ArrayList<Integer> l, int length){
        locations=new ArrayList<>(l);    
        intervalLength=length;
        baseShape.setLength(intervalLength);
    }
    public ArrayList<Integer> getIntervals(){return locations;}
    public int getIntervalLength(){ return intervalLength;}
    
    public boolean setNonOverlappingIntervals(){
//Me giving up and just randomly placing the shapes until they are all non overlapping
        for(int i=0;i<nosIntervals;i++){
            boolean ok=false;
            int l=intervalLength/2;
            while(!ok){
                ok=true;
//Search mid points to level the distribution up somewhat                
                l=rand.nextInt(seriesLength-intervalLength)+intervalLength/2;
  //          System.out.println("trying   "+l);
                
                for(int in:locations){
//I think this is setting them too big                    
                    if((l>=in-intervalLength && l<in+intervalLength) //l inside ins
                      ||(l<in-intervalLength && l+intervalLength>in)      ){ //ins inside l
                        ok=false;
//                        System.out.println(l+"  overlaps with "+in);
                        break;
                    }
                }
            }
//           System.out.println("Adding "+l);
            locations.add(l);
        }
//Revert to start points            
        for(int i=0;i<locations.size();i++){
            int val=locations.get(i);
            locations.set(i, val-intervalLength/2);
        }
        Collections.sort(locations);
        return true;
    }

    @Override
    public double generate(){
//Noise
        double value=error.simulate();
        int insertionPoint=0;
        while(insertionPoint<locations.size() && locations.get(insertionPoint)+intervalLength<t)
            insertionPoint++;    
        if(insertionPoint>=locations.size()){ //Bigger than all the start points, set to last
            insertionPoint=locations.size()-1;
        }
        int point=locations.get(insertionPoint);
        if(point<=t && point+intervalLength>t){//in shape1
            value+=baseShape.generateWithinShapelet((int)(t-point));
//                System.out.println(" IN SHAPE 1 occurence "+insertionPoint+" Time "+t);
        }
        t++;
        return value;
    }
    
    
    @Override
    public void setParameters(double[] p) {
        nosIntervals=(int)p[0];        
        intervalLength=(int)p[1];        
    }
    public void randomiseShape(IntervalModel m){
        baseShape.randomiseShape();
        if(m!=null){
            while(baseShape.equals(m.baseShape))
                baseShape.randomiseShape();
        }
        
    }
    public static void main(String[] args){
//Set up two models with same intervals but different shapes        
        int length=500;
        Model.setGlobalRandomSeed(10);
        Model.setDefaultSigma(0);
        IntervalModel m1=new IntervalModel();
        m1.setBaseShapeType(ShapeType.SINE);
        m1.setNosIntervals(3);
        m1.setSeriesLength(length);
        m1.createIntervals();
        IntervalModel m2=new IntervalModel();
        m2.setBaseShapeType(ShapeType.SPIKE);
        m2.setNosIntervals(3);
        m2.setIntervals(m1.getIntervals(), m1.getIntervalLength());
        double[] d1=m1.generateSeries(length);
        double[] d2=m2.generateSeries(length);
        OutFile of=new OutFile("C:\\temp\\intervalEx.csv");
        for(int i=0;i<length;i++)
            of.writeLine(d1[i]+","+d2[i]);
    }
    @Override
    public String toString(){
        String s="NosIntervals,"+nosIntervals;
        s+="\nIntervalLength,"+intervalLength;
        s+="\nNoiseToSignal,"+noiseToSignal;
        s+="\nShape,"+baseShape.toString()+" ,Locations,";
        for(int i=0;i<nosIntervals;i++)
            s+=locations.get(i)+" ";
        return s;
    }
    
}
