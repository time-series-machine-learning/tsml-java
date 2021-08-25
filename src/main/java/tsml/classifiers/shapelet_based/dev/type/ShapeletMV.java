package tsml.classifiers.shapelet_based.dev.type;

import tsml.data_containers.TimeSeriesInstance;
import utilities.rescalers.ZNormalisation;

public abstract class ShapeletMV implements Comparable<ShapeletMV>{

    protected static ZNormalisation NORMALIZE = new ZNormalisation();

    protected int length;
    protected double classIndex;
    protected double quality;

    public ShapeletMV(int length, double classIndex){
        this.length = length;
        this.classIndex = classIndex;
    }

    public void setQuality(double quality){
        this.quality = quality;
    }
    public double getQuality(){
        return this.quality;
    }
    public double getClassIndex() {
        return classIndex;
    }
    public void setClassIndex(double classIndex) {
        this.classIndex = classIndex;
    }
    public int getLength(){
        return length;
    }

    protected abstract void setData(TimeSeriesInstance instance);


    @Override
    public int compareTo(ShapeletMV shapeletMV) {
        return (this.getQuality()>shapeletMV.getQuality()?
                -1:
                (this.getQuality()<shapeletMV.getQuality()?
                        1:
                        (this.getLength()>shapeletMV.getLength()?
                                1:
                                (this.getLength()<shapeletMV.getLength()?-1:0))));
    }
}
