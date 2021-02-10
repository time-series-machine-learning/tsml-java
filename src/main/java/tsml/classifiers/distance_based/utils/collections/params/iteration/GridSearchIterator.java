/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.utils.collections.params.iteration;

import java.io.Serializable;
import java.util.Iterator;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.IndexedParameterSpace;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class GridSearchIterator implements Iterator<ParamSet>, Serializable {

    private IndexedParameterSpace indexedParameterSpace;
    private int iterationCount = 0;
    private int numIterations = -1;

    public GridSearchIterator(final ParamSpace paramSpace) {
        setParameterSpace(paramSpace);
    }

    public ParamSpace getParameterSpace() {
        return indexedParameterSpace.getParamSpace();
    }

    public IndexedParameterSpace getIndexedParameterSpace() {
        return indexedParameterSpace;
    }

    protected void setParameterSpace(
        final ParamSpace paramSpace) {
        indexedParameterSpace = new IndexedParameterSpace(paramSpace);
        numIterations = indexedParameterSpace.size();
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "{iterationCount=" + iterationCount + ", parameterSpace=" + indexedParameterSpace.getParamSpace().toString() + "}";
    }

    @Override
    public boolean hasNext() {
        return iterationCount < numIterations;
    }

    @Override
    public ParamSet next() {
        ParamSet paramSet = getIndexedParameterSpace().get(iterationCount);
        iterationCount++;
        return paramSet;
    }

    public int getNumIterations() {
        return numIterations;
    }

    public void setNumIterations(final int numIterations) {
        Assert.assertTrue(numIterations >= 0);
        this.numIterations = numIterations;
    }
}
