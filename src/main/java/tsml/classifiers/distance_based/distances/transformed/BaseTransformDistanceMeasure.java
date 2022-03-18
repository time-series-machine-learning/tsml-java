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
 
package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.ed.EDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeriesInstance;
import tsml.transformers.Transformer;
import weka.core.Instance;

import java.util.Objects;

public class BaseTransformDistanceMeasure extends BaseDistanceMeasure implements TransformDistanceMeasure {

    public BaseTransformDistanceMeasure(String name, Transformer transformer,
                                        DistanceMeasure distanceMeasure) {
        setDistanceMeasure(distanceMeasure);
        setTransformer(transformer);
        setName(name);
    }

    public BaseTransformDistanceMeasure() {
        this(null, null, new EDistance());
    }
    
    public BaseTransformDistanceMeasure(String name) {
        this(name, null, new EDistance());
    }

    public static final String TRANSFORMER_FLAG = "t";
    private DistanceMeasure distanceMeasure;
    private Transformer transformer;
    private String name;

    @Override public String getName() {
        return name;
    }

    public void setName(String name) {
        if(name == null) {
            name = distanceMeasure.getName();
            if(transformer != null) {
                name = transformer.getClass().getSimpleName() + "_" + name;
            }
        }
        this.name = name;
    }

    @Override public boolean isSymmetric() {
        return distanceMeasure.isSymmetric();
    }
    
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        return transform(inst, true);
    }
    
    public TimeSeriesInstance transform(TimeSeriesInstance inst, boolean transform) {
        if(transform) {
            if(transformer != null) {
                inst = transformer.transform(inst);
            }
        }
        return inst;
    }
    
    public double distance(final TimeSeriesInstance a, final boolean transformA, final TimeSeriesInstance b, final boolean transformB, final double limit) {
        try {
            // users must call fit method on transformer if required before calling distance
            final TimeSeriesInstance at = transform(a, transformA);
            // need to take the interval here, before the transform
            final TimeSeriesInstance bt = transform(b, transformB);
            return distanceMeasure.distance(at, bt, limit);
        } catch(Exception e) {
            throw new IllegalStateException(e);
        }
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(DistanceMeasure distanceMeasure) {
        this.distanceMeasure = Objects.requireNonNull(distanceMeasure);
    }

    @Override public ParamSet getParams() {
        final ParamSet paramSet = super.getParams();
        paramSet.add(TRANSFORMER_FLAG, transformer);
        paramSet.add(DistanceMeasure.DISTANCE_MEASURE_FLAG, distanceMeasure);
        return paramSet;
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        setTransformer(param.get(TRANSFORMER_FLAG, transformer));
        setDistanceMeasure(param.get(DISTANCE_MEASURE_FLAG, distanceMeasure));
        super.setParams(param);
    }

    public Transformer getTransformer() {
        return transformer;
    }

    public void setTransformer(Transformer a) {
        transformer = a;
    }

    @Override public String toString() {
        return getName() + " " + distanceMeasure.getParams();
    }
}
