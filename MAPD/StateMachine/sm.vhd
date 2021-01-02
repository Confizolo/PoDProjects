LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
-- entity
ENTITY my_fsm1 IS
    PORT (
        TOG_EN, CLK, CLR : IN STD_LOGIC;
        Z1
        : OUT STD_LOGIC);
END my_fsm1;
ARCHITECTURE fsm1 OF my_fsm1 IS
    TYPE state_type IS (ST0, ST1);
    SIGNAL state : state_type;
BEGIN
    sync_proc : PROCESS (CLK)
    BEGIN
        IF (rising_edge(CLK)) THEN
            IF (CLR = '1') THEN
                state <= ST0;
                Z1
                <= '0';
                -- pre-assign
            ELSE
                CASE state IS
                    WHEN ST0 => -- items regarding state ST0 Z1 <= '0'; -- Moore output
                        Z1
                        <= '0'; -- pre-assign
                        IF (TOG_EN = '1') THEN
                            state <= ST1;
                        END IF;
                    WHEN ST1 =>
                        -- items regarding state ST1
                        Z1
                        <= '1'; -- Moore output
                        IF (TOG_EN = '1') THEN
                            state <= ST0;
                        END IF;
                    WHEN OTHERS =>
                        -- the catch-all condition
                        Z1
                        <= '0';
                        -- arbitrary; it should never
                        state <= ST0;
                        -- make it to these two statements
                END CASE;
            END IF;
        END IF;
    END PROCESS sync_proc;
END fsm1;