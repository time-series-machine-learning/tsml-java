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
package utilities;

import java.util.function.Consumer;

/**
 * Just saves some time/copied code when making new classes
 * 
 * And in general the codebase needed a bit more hacky-ness to it 
 * 
 * @author James Large
 */
public interface DebugPrinting {
    
    final static Consumer<String> printer = (s) -> System.out.print(s);
    final static Consumer<String> printlner = (s) -> System.out.println(s);
    final static Consumer<String> nothing_placeholder = (s) ->  { };
    
    //defaults to printing nothing
    static Consumer[] printers = new Consumer[] { nothing_placeholder, nothing_placeholder };
    
    default void setDebugPrinting(boolean b) {
        if (b) {
            printers[0] = printer;
            printers[1] = printlner;
        }
        else {
            printers[0] = nothing_placeholder;
            printers[1] = nothing_placeholder;
        }
    }
    
    default boolean getDebugPrinting() {
        return printers[0] == nothing_placeholder;
    }
    
    default void printDebug(String str) {
        printers[0].accept(str);
    }
    
    default void printlnDebug(String str) {
        printers[1].accept(str);
    }
}
