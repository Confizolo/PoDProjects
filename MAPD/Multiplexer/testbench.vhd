LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT mux4 IS
        PORT (
            a1, a2, a3, a4 : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
            sel
            : IN STD_LOGIC_VECTOR(1 DOWNTO 0);
            b
            : OUT STD_LOGIC_VECTOR(2 DOWNTO 0));
    END COMPONENT;
    SIGNAL a1, a2, a3, a4, b : STD_LOGIC_VECTOR(2 DOWNTO 0);
    SIGNAL sel : STD_LOGIC_VECTOR(1 DOWNTO 0);
BEGIN
    DUT : mux4 PORT MAP(a1, a2, a3, a4, sel, b);
    PROCESS
    BEGIN
        a1 <= B"101";
        a2 <= B"010";
        a3 <= B"010";
        a4 <= B"010";
        sel <= B"10";
        WAIT FOR 1 ns;
        a1 <= B"101";
        a2 <= B"110";
        a3 <= B"110";
        a4 <= B"101";
        sel <= B"11";
        WAIT FOR 1 ns;
        WAIT;
    END PROCESS;
END tb;