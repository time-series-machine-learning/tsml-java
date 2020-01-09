/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */   
package tsml.transformers.shapelet_tools;

import java.io.Serializable;

 /**
     * copyright: Anthony Bagnall
     * A simple class to store <distance,classValue> pairs for calculating the quality of a shapelet.
     */
    public final class OrderLineObj implements Comparable<OrderLineObj>, Serializable {

        private double distance;
        private double classVal;
        /**
         * Constructor to build an orderline object with a given distance and class value
         * @param distance distance from the obj to the shapelet that is being assessed
         * @param classVal the class value of the object that is represented by this OrderLineObj
         */
        public OrderLineObj(double distance, double classVal){
            this.distance = distance;
            this.classVal = classVal;
        }

        /**
         * Accessor for the distance field
         * @return this OrderLineObj's distance
         */
        public double getDistance(){
            return this.distance;
        }

        /**
         * Accessor for the class value field of the object
         * @return this OrderLineObj's class value
         */
        public double getClassVal(){
            return this.classVal;
        }

        /**
         * Mutator for the distance field
         * @param distance new distance for this OrderLineObj
         */
        public void setDistance(double distance){
            this.distance = distance;
        }
        
        /**
         * Mutator for the class value field of the object
         * @param classVal new class value for this OrderLineObj
         */
        public void setClassVal(double classVal){
            this.classVal = classVal;
        }
        
        /**
         * Comparator for two OrderLineObj objects, used when sorting an orderline
         * @param o the comparison OrderLineObj
         * @return the order of this compared to o: -1 if less, 0 if even, and 1 if greater.
         */
        @Override
        public int compareTo(OrderLineObj o) {
            // return distance - o.distance. compareTo doesnt care if its -1 or -inf. likewise +1 or +inf.  
            if(o.distance > this.distance){
                return -1;
            }else if(o.distance==this.distance){
                return 0;
            }
            return 1;
        }
        
        @Override
        public String toString()
        {
            return distance + "," + classVal;
        }
    }

