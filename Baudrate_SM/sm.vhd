LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
-- entity
ENTITY my_fsm IS
    PORT (
        DATA_EN : IN STD_LOGIC_VECTOR(0 TO 7);
        CLK, VALID : IN STD_LOGIC;
        UARTtx, BUSY
        : OUT STD_LOGIC);
END my_fsm;
ARCHITECTURE fsm OF my_fsm IS
    TYPE state_type IS (IDLE, RD0, RD1, RD2, RD3, RD4, RD5, RD6, RD7);
    SIGNAL state : state_type;
BEGIN
    sync_proc : PROCESS (CLK)
    BEGIN
        IF (rising_edge(CLK)) THEN

            CASE state IS
                WHEN IDLE => 
                    BUSY
                    <= '0';
                    IF VALID = '1' THEN
                        state <= RD0;
                    END IF;
                WHEN RD0 =>
                    BUSY
                    <= '1'; 
                    UARTtx <= DATA_EN(0);
                    state <= RD1;
                WHEN RD1 =>
                    BUSY
                    <= '1'; 
                    UARTtx <= DATA_EN(1);
                    state <= RD2;
                WHEN RD2 =>
                    
                    BUSY
                    <= '1'; -- Moore output
                    UARTtx <= DATA_EN(2);
                    state <= RD3;
                WHEN RD3 =>
                    -- items regarding state ST1
                    BUSY
                    <= '1'; -- Moore output
                    UARTtx <= DATA_EN(3);
                    state <= RD4;
                WHEN RD4 =>
                    -- items regarding state ST1
                    BUSY
                    <= '1'; -- Moore output
                    UARTtx <= DATA_EN(4);
                    state <= RD5;
                WHEN RD5 =>
                    -- items regarding state ST1
                    BUSY
                    <= '1'; -- Moore output
                    UARTtx <= DATA_EN(5);
                    state <= RD6;
                WHEN RD6 =>
                    -- items regarding state ST1
                    BUSY
                    <= '1'; -- Moore output
                    UARTtx <= DATA_EN(6);
                    state <= RD7;
                WHEN RD7 =>
                    -- items regarding state ST1
                    BUSY
                    <= '1'; -- Moore output
                    UARTtx <= DATA_EN(7);
                    state <= IDLE;
                WHEN OTHERS =>
                    -- the catch-all condition
                    BUSY
                    <= '0';
                    -- arbitrary; it should never
                    state <= IDLE;
                    -- make it to these two statements
            END CASE;
        END IF;
    END PROCESS sync_proc;
END fsm;