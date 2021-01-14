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
 
package tsml.classifiers.distance_based.knn;

import org.junit.Assert;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import weka.core.Instances;

import java.util.Iterator;

public class KNNTuningAgent extends BaseTuningAgent {

    private ParamSpace paramSpace;
    private ParamSpaceBuilder paramSpaceBuilder;
    private ParamSetIteratorBuilder paramSetIteratorBuilder;
    private Iterator<ParamSet> paramSetIterator;
    private ClassifierBuilder classifierBuilder;

    public ClassifierBuilder getClassifierBuilder() {
        return classifierBuilder;
    }

    public void setClassifierBuilder(
            final ClassifierBuilder classifierBuilder) {
        this.classifierBuilder = classifierBuilder;
    }

    public interface ParamSetIteratorBuilder {
        Iterator<ParamSet> build(ParamSpace space);
    }

    public interface ClassifierBuilder {
        EnhancedAbstractClassifier build();
    }

    @Override public void buildAgent(final Instances trainData) {
        Assert.assertNotNull(paramSetIteratorBuilder);
        Assert.assertNotNull(classifierBuilder);
        Assert.assertFalse(paramSpace == null && paramSpaceBuilder == null);
        super.buildAgent(trainData);
        if(paramSpaceBuilder != null) {
            paramSpace = paramSpaceBuilder.build(trainData);
        }
        paramSetIterator = paramSetIteratorBuilder.build(paramSpace);
    }

    @Override protected Benchmark nextExplore() {
        final EnhancedAbstractClassifier classifier = classifierBuilder.build();
        final ParamSet paramSet = paramSetIterator.next();
        if(classifier instanceof ParamHandler) {
            try {
                ((ParamHandler) classifier).setParams(paramSet);
            } catch(Exception e) {
                throw new IllegalStateException(e);
            }
        } else {
            try {
                classifier.setOptions(paramSet.getOptions());
            } catch(Exception e) {
                throw new IllegalStateException(e);
            }
        }
        return new Benchmark(classifier);
    }

    @Override protected Benchmark nextExploit() {
        throw new UnsupportedOperationException();
    }

    @Override protected boolean hasNextExplore() {
        return paramSetIterator.hasNext();
    }

    @Override protected boolean hasNextExploit() {
        throw new UnsupportedOperationException();
    }

    @Override protected boolean shouldExplore() {
        throw new UnsupportedOperationException();
    }

    @Override protected boolean isExploitable(final Benchmark benchmark) {
        final EnhancedAbstractClassifier classifier = benchmark.getClassifier();
        if(!(classifier instanceof KNN)) {
            throw new IllegalStateException("expected only KNN");
        }
        throw new UnsupportedOperationException();
    }

    public ParamSpace getParamSpace() {
        return paramSpace;
    }

    public void setParamSpace(final ParamSpace paramSpace) {
        this.paramSpace = paramSpace;
    }

    public ParamSpaceBuilder getParamSpaceBuilder() {
        return paramSpaceBuilder;
    }

    public void setParamSpaceBuilder(final ParamSpaceBuilder paramSpaceBuilder) {
        this.paramSpaceBuilder = paramSpaceBuilder;
    }

    public ParamSetIteratorBuilder getParamSetIteratorBuilder() {
        return paramSetIteratorBuilder;
    }

    public void setParamSetIteratorBuilder(
            final ParamSetIteratorBuilder paramSetIteratorBuilder) {
        this.paramSetIteratorBuilder = paramSetIteratorBuilder;
    }

}
