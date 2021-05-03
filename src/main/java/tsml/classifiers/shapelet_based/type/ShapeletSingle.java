package tsml.classifiers.shapelet_based.type;

public abstract class ShapeletSingle extends ShapeletMV{

    protected int start;

    protected int instanceIndex;




    public ShapeletSingle(int start, int length, int instanceIndex, double classIndex){
        super(length,classIndex);
        this.start = start;
        this.instanceIndex = instanceIndex;
    }



}
