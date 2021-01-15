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
 
package tsml.classifiers.distance_based.utils.classifiers;

import weka.classifiers.Classifier;

/**
 * Purpose: factory to produce classifier builders. Anything added to this factory is checked to ensure there is a
 * corresponding final field in the factory by the same name as the builder. E.g.
 * public static class Factory {
 *     public static final MY_CLSF = add(<classifier builder with the name MY_CLSF);
 * }
 * <p>
 * Contributors: goastler
 */
public class CompileTimeClassifierBuilderFactory<B extends Classifier> extends ClassifierBuilderFactory<B> {

//    public ClassifierBuilder<? extends B> add(ClassifierBuilder<? extends B> builder) {
//        try {
//            // must have field matching builder's name
//            Field field = this.getClass().getDeclaredField(builder.getName());
//            if(!Modifier.isFinal(field.getModifiers())) {
//                throw new IllegalStateException(builder.getName() + "field not final");
//            }
//        } catch(NoSuchFieldException e) {
//            throw new IllegalStateException(e);
//        }
//        super.add(builder);
//        return builder;
//    }

}
