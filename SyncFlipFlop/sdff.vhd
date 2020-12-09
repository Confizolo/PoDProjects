LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY sdff IS
    PORT (
        clk: IN std_logic;
        rst1, rst2, d1 : IN STD_LOGIC;
        q2 : OUT STD_LOGIC
    );
END sdff;
ARCHITECTURE rtl OF sdff IS
    COMPONENT dff IS
        PORT (
            clk: IN std_logic;
            d, rst : IN STD_LOGIC;
            q : OUT STD_LOGIC
        );
    END COMPONENT dff;
    SIGNAL dint : STD_LOGIC;
BEGIN
    dff1 : dff
    PORT MAP(
        clk =>clk,
        d => d1,
        q => dint,
        rst => rst1
    );
    dff2 : dff
    PORT MAP(
        clk => clk,
        q => q2,
        d =>dint,
        rst => rst2
    );
END rtl;
