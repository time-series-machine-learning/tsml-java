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
 
package tsml.classifiers.distance_based.utils.collections.params;

import org.apache.commons.collections4.iterators.TransformIterator;
import weka.classifiers.Classifier;

import java.util.Iterator;
import java.util.function.Supplier;

/**
 * Purpose: Iterate over several ParamSets whilst building a corresponding classifier for each one.
 *
 * Contributors: goastler
 */
public class ParamSetClassifierIterator extends TransformIterator<ParamSet, Classifier> {
    private Supplier<Classifier> classifierBuilder;

    public ParamSetClassifierIterator(final Iterator<ParamSet> paramSetIterator,
                                      final Supplier<Classifier> classifierBuilder) {
        super(paramSetIterator);
        setClassifierBuilder(classifierBuilder);
    }

    public Supplier<Classifier> getClassifierBuilder() {
        return classifierBuilder;
    }

    public void setClassifierBuilder(final Supplier<Classifier> classifierBuilder) {
        this.classifierBuilder = classifierBuilder;
        setTransformer(paramSet -> {
            Classifier classifier = classifierBuilder.get();
            ParamHandler.setParams(classifier, paramSet);
            return classifier;
        });
    }
}
