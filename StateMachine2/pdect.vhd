LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY patterndetect IS
    PORT (
        a
        : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        rst : IN STD_LOGIC;
        y
        : OUT STD_LOGIC);
END ENTITY patterndetect;
ARCHITECTURE rtl OF patterndetect IS
    TYPE state_t IS (S0, S1, S2, S3, Detect);
    SIGNAL state : state_t := S0;
BEGIN -- architecture rtl
    main : PROCESS (clk) IS
    BEGIN -- process main
        IF rising_edge(clk) THEN
            -- rising clock edge
            IF rst = '0' THEN
                -- synchronous reset (active low)
                state <= S0;
                y
                <= '0';
            ELSE
                CASE state IS
                    WHEN S0 =>
                        y <= '0';
                        IF a = '0' THEN
                            state <= S1;
                        END IF;
                    WHEN S1 =>
                        y <= '0';
                        IF a = '0' THEN
                            state <= S1;
                        ELSIF a = '1' THEN
                            state <= S2;
                            END IF;
                        WHEN S2 =>
                            y <= '0';
                            IF a = '0' THEN
                                state <= S3;
                            ELSIF a = '1' THEN
                                state <= S0;
                            ELSE
                                NULL;
                            END IF;
                        WHEN S3 =>
                            y <= '0';
                            IF a = '0' THEN
                                state <= S1;
                            ELSIF a = '1' THEN
                                state <= Detect;
                            ELSE
                                NULL;
                            END IF;
                        WHEN Detect =>
                            y
                            <= '1';
                            state <= S0;
                        WHEN OTHERS => NULL;
                        END CASE;
                END IF;
            END IF;
        END PROCESS main;
    END ARCHITECTURE rtl;