/*

Elastic model for simulators.


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
public class ElasticModel extends Model{
    private static double warpPercent=0.1;
    private int seriesLength=200; // Need to set intervals, maybe allow different lengths? 
    private static double base=-1;
    private static double amplitude=2;
    private Shape baseShape;
    int start =0;
    int end=seriesLength;
    public ElasticModel(){
        baseShape=new Shape();
        Shape.DEFAULTAMP=amplitude;
        Shape.DEFAULTBASE=base;
    }
    public void setSeriesLength(int n){ 
        seriesLength=n;
    }
    public void setBaseShapeType(ShapeType st){
        baseShape.setType(st);
    }
    public static void setWarpPercent(double b){
        warpPercent=b;
    }
    public static void setBaseAndAmp(double b, double a){
        amplitude=a;
        base=b;
        Shape.DEFAULTAMP=amplitude;
        Shape.DEFAULTBASE=base;
    }
 
    @Override
    public double generate(){
//Noise
        double value=error.simulate();
        if(t>=start && t<end)
            value+=baseShape.generateWithinShapelet((int)(t-start));
        t++;
        return value;
    }
    @Override
    public double[] generateSeries(int n){
//Set random start and end and set up shape length
        start=Model.rand.nextInt((int)(warpPercent*seriesLength));
        end=seriesLength-Model.rand.nextInt((int)(warpPercent*seriesLength));
 //       -Model.rand.nextInt((int)(warpPercent*seriesLength+1));
        baseShape.setLength(end-start);
//        System.out.println("Length ="+seriesLength+" Start =  "+start+" End =  "+end+"  Shape Length ="+(end-start));
        double[] series=new double[n];
        t=0;
        for(int i=0;i<n;i++)
            series[i]=generate();
        return series;
    }
    @Override
    public void setParameters(double[] p) {
        warpPercent=p[0];
        
    }
    public void randomiseShape(DictionaryModel.ShapeType m){
        baseShape.randomiseShape();
        if(m!=null){
            while(baseShape.type==m)
                baseShape.randomiseShape();
        }        
    }
    public void randomiseShape(ElasticModel m){
        baseShape.randomiseShape();
        if(m!=null){
            while(baseShape.equals(m.baseShape))
                baseShape.randomiseShape();
        }        
    }
    public DictionaryModel.ShapeType getShape(){
        return baseShape.type;
    }    
    public void setShape(DictionaryModel.ShapeType sh){
        baseShape.type=sh;
    }    
    public static void main(String[] args){
    }
    @Override
    public String toString(){
        return baseShape.toString()+" warp ="+warpPercent;
    }
    
}
