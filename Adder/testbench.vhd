LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY adder_tb IS
END ENTITY adder_tb;
ARCHITECTURE test OF adder_tb IS
    CONSTANT width : INTEGER := 8;
    SIGNAL a
    : STD_LOGIC_VECTOR(width - 1 DOWNTO 0);
    SIGNAL b
    : STD_LOGIC_VECTOR(width - 1 DOWNTO 0);
    SIGNAL sum : STD_LOGIC_VECTOR(width - 1 DOWNTO 0);
    SIGNAL Clk : STD_LOGIC := '1';
BEGIN -- architecture test
    DUT : ENTITY work.adder
        GENERIC MAP(
            width => width)
        PORT MAP(
            a
            => a,
            b
            => b,
            sum => sum);
    main : PROCESS IS
    BEGIN -- process main
        a <= X"AA";
        B <= X"BB";
        WAIT FOR 1 ns;
        a <= X"A0";
        B <= X"B0";
        WAIT FOR 1 ns;
        WAIT;
    END PROCESS main;
END ARCHITECTURE test;