LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY generator_1 IS
    PORT (
        clk : IN STD_LOGIC;
        baud_out : OUT STD_LOGIC
    );
END generator_1;
ARCHITECTURE rtl OF generator_1 IS
    SIGNAL counter : unsigned(9 DOWNTO 0) := (OTHERS => '0');
    CONSTANT divisor : unsigned(9 DOWNTO 0) := to_unsigned(867, 10);
BEGIN -- architecture rtl
    main : PROCESS (clk) IS
    BEGIN -- process main
        IF rising_edge(clk) THEN
            counter <= counter + 1;
            IF counter = divisor THEN
                baud_out <= '1';
                counter <= (OTHERS => '0');
            ELSE
                baud_out <= '0';
            END IF;
        END IF;
    END PROCESS main;
END ARCHITECTURE rtl;