package tsml.classifiers.distance_based.utils.collections.params.iteration;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;

import java.io.Serializable;
import java.util.Iterator;

public abstract class ParamSpaceSearch implements Iterator<ParamSet>, Serializable {
    
    private ParamSpace paramSpace;
    private boolean hasNextCalled;
    private boolean hasNext;
    private int iterationCount;
    
    @Override
    public String toString() {
        return getClass().getSimpleName();
    }

    public void buildSearch(ParamSpace paramSpace) {
        Assert.assertNotNull(paramSpace);
        this.paramSpace = paramSpace;
        hasNextCalled = false;
        iterationCount = 0;
    }

    public ParamSpace getParamSpace() {
        return paramSpace;
    }
    
    protected abstract boolean hasNextParamSet();
    
    protected abstract ParamSet nextParamSet();
    
    @Override public final boolean hasNext() {
        if(hasNextCalled) {
            return hasNext;
        }
        return hasNext = hasNextParamSet();
    }

    @Override public final ParamSet next() {
        if(!hasNext) {
            throw new IllegalStateException("hasNext false");
        }
        iterationCount++;
        return nextParamSet();
    }

    public int getIterationCount() {
        return iterationCount;
    }
}
