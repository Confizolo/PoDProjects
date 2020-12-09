LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY mux2 IS
    PORT (
        a1 : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
        a2 : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
        sel : IN STD_LOGIC;
        b
        : OUT STD_LOGIC_VECTOR(2 DOWNTO 0));
END mux2;
ARCHITECTURE rtl OF mux2 IS
BEGIN
    p_mux : PROCESS (a1, a2, sel)
    BEGIN
        CASE sel IS
            WHEN '0'
                => b <= a1;
            WHEN '1'
                => b <= a2;
            WHEN OTHERS => b <= "000";
        END CASE;
    END PROCESS p_mux;
END rtl;