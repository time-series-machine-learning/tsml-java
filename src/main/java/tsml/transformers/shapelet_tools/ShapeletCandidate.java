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
 *
 * @author raj09hxu
 */
public class ShapeletCandidate implements Serializable{
    double[][] content;
    int numChannels;
    
    //if no dimension, assume univariate
    public ShapeletCandidate(){
        numChannels = 1;
        content = new double[numChannels][];
    }
    
    public ShapeletCandidate(int numChans){
        numChannels = numChans;
        content = new double[numChannels][];
    }
    
    public ShapeletCandidate(double[] cont){
        numChannels = 1;
        content = new double[numChannels][];
        content[0] = cont;
    }
    
    //if no dimension, assume univariate
    public void setShapeletContent(double[] cont){
        content[0] = cont;
    }
    
    public void setShapeletContent(int channel, double[] cont){
        content[channel] = cont;
    }

    public double[] getShapeletContent(int channel){
        return content[channel]; 
    }
    
    public double[] getShapeletContent(){
        return content[0];
    }
    
    public int getLength(){
        return content[0].length;
    }
    
    public int getNumChannels(){
        return numChannels;
    }
}
